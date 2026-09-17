"""SheetSage2's infer.py with its audio encoder on MLX.

    .venv-sheetsage/bin/python sheetsage_runner.py <infer.py arguments> --mlx-encoder <dir>

Everything infer.py does is unchanged -- windowing, the grammar-constrained
decoder, every export -- except where the encoder runs. SheetSage2 normally
loads its 2.4GB MERT-v2 parent in PyTorch and merges LoRA adapters into it; the
converted encoder (mlx_sheetsage.py) already carries the merged weights, so the
parent is never read and may be deleted. The decoder keeps running wherever
--device puts it.

Lives outside sheetsage2/ because setup_sheetsage.sh downloads that directory
again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import mlx_sheetsage

SHEETSAGE = Path(__file__).resolve().parent / "sheetsage2"
MEL_CHUNK = 2048   # STFT frames per chunk (of 30001 in a window)


class ParentStub(torch.nn.Module):
    """What SheetSage2 needs of MERT-v2 when MLX does the encoding: config and front end.

    No Conformer blocks, so merge_lora has nothing to merge into and only marks
    the model merged.
    """

    layers = ()

    def __init__(self, mert, config, mel_mean, mel_std):
        super().__init__()
        self.config = config
        self.feature_extractor = mert.MERT2MelFrontend(config)
        self.feature_extractor.mel_mean.copy_(torch.tensor(mel_mean))
        self.feature_extractor.mel_std.copy_(torch.tensor(mel_std))
        self.feature_extractor.forward = self.log_mel

    @torch.no_grad()
    def log_mel(self, waveform):
        """MERT2MelFrontend.forward, computed MEL_CHUNK frames at a time.

        The whole-window STFT is a 30001 x 1025 complex tensor, and the CPU
        allocator kept the ~0.9 GB it passed through for the rest of the run.
        Frames are independent once the waveform carries its centre padding, so
        framing the padded signal chunk by chunk gives the same values --
        checked bit-identical against the unchunked front end.
        """
        frontend = self.feature_extractor
        spectrogram = frontend.spectrogram
        n_fft, hop = spectrogram.n_fft, spectrogram.hop_length
        waveform = waveform.float()
        padded = torch.nn.functional.pad(waveform[:, None], (n_fft // 2, n_fft // 2),
                                         mode=spectrogram.pad_mode)[:, 0]
        frames = 1 + (padded.shape[-1] - n_fft) // hop
        chunks = []
        for start in range(0, frames, MEL_CHUNK):
            count = min(MEL_CHUNK, frames - start)
            piece = padded[:, start * hop:(start + count - 1) * hop + n_fft]
            power = torch.stft(piece, n_fft, hop, spectrogram.win_length, spectrogram.window,
                               center=False, onesided=spectrogram.onesided,
                               return_complex=True).abs().pow(spectrogram.power)
            chunks.append(frontend.amplitude_to_db(frontend.mel_scale(power)))
        mel = torch.cat(chunks, dim=-1)[..., :-1].transpose(-1, -2)
        return (mel - frontend.mel_mean) / frontend.mel_std.clamp_min(1e-5)


def answer_parent_check(modeling, checkpoint, sheetsage, encoder):
    """SheetSage2's MERT-v2 integrity check, answered without downloading the parent.

    It hashes the parent's two code files and its weights before loading. The
    code files ship byte-identical inside the SheetSage2 checkpoint, so those are
    hashed for real; the weights are vouched for by the digest mlx_sheetsage.py
    recorded when it converted them.
    """
    parent = sheetsage["base_model_name_or_path"]
    stand_in = str(Path(encoder) / mlx_sheetsage.WEIGHTS)
    fetch, digest = modeling.cached_file, modeling._sha256

    def cached_file(repository, filename, **kwargs):
        if repository == parent:
            if filename == "model.safetensors":
                return stand_in
            if (Path(checkpoint) / filename).is_file():
                return str(Path(checkpoint) / filename)
        return fetch(repository, filename, **kwargs)

    def sha256(path):
        return sheetsage["base_model_sha256"] if str(path) == stand_in else digest(path)

    modeling.cached_file, modeling._sha256 = cached_file, sha256


def use_mlx_encoder(directory):
    from transformers import AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    weights, config = mlx_sheetsage.load(directory)
    # Freed activations would otherwise sit in MLX's buffer cache (1.1 GB after
    # one window) while the decoder runs.
    mlx_sheetsage.mx.set_cache_limit(0)
    load = AutoModel.from_pretrained
    loading = []   # (package, SheetSage2 config) while its parent is being fetched

    def get_audio_features(self, input_values, attention_mask=None, output_hidden_states=None, return_dict=True):
        if output_hidden_states:
            raise ValueError("The MLX encoder does not export per-block backbone states; drop --all-layers.")
        waveform, lengths = self._prepare_audio(input_values, attention_mask)
        mel = self.encoder.feature_extractor(waveform.float().cpu())
        memory = torch.from_numpy(mlx_sheetsage.encode(weights, config, mel.numpy())).to(waveform.device)
        stride = self.encoder.config.inputs_to_logits_ratio
        frames = torch.arange(memory.shape[1], device=memory.device)[None]
        mask = frames < ((lengths.to(memory.device) + stride - 1) // stride)[:, None]
        output = sys.modules[type(self).__module__].SheetSage2EncoderOutput(
            encoder_last_hidden_state=memory, feature_attention_mask=mask)
        return output if return_dict else output.to_tuple()

    def from_pretrained(path, *args, **kwargs):
        if loading:
            # The nested call is SheetSage2 fetching its MERT-v2 parent.
            package, sheetsage = loading[-1]
            mert_config = sys.modules[f"{package}.configuration_mert2"].MERT2Config(**sheetsage["backbone_config"])
            return ParentStub(sys.modules[f"{package}.modeling_mert2"], mert_config,
                              config["mel_mean"], config["mel_std"])
        sheetsage = json.loads((Path(path) / "config.json").read_text())
        if sheetsage["base_model_sha256"] != config["source"]["base_model_sha256"]:
            raise ValueError("The MLX encoder was converted from a different MERT-v2 parent; "
                             "run mlx_sheetsage.py convert again.")
        model_class = get_class_from_dynamic_module(sheetsage["auto_map"]["AutoModel"], path)
        modeling = sys.modules[model_class.__module__]
        answer_parent_check(modeling, path, sheetsage, directory)
        loading.append((model_class.__module__.rpartition(".")[0], sheetsage))
        try:
            model = load(path, *args, **kwargs)
        finally:
            loading.pop()
        model.get_audio_features = get_audio_features.__get__(model)
        return model

    AutoModel.from_pretrained = staticmethod(from_pretrained)


def main():
    arguments = sys.argv[1:]
    directory = mlx_sheetsage.DEFAULT_DIR
    if "--mlx-encoder" in arguments:
        index = arguments.index("--mlx-encoder")
        directory = Path(arguments[index + 1])
        del arguments[index:index + 2]
    use_mlx_encoder(directory)
    sys.path.insert(0, str(SHEETSAGE))
    import infer
    sys.argv = [str(SHEETSAGE / "infer.py"), *arguments]
    infer.main()


if __name__ == "__main__":
    main()
