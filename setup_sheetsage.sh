#!/bin/bash
# SheetSage2 pins torch 2.8 / transformers 4.45 / numpy 1.24, which conflict with
# yue2's torch 2.10 / transformers 4.57. It therefore gets its own environment and
# is driven as a subprocess.
set -e
cd "$(dirname "$0")"
uv venv --python 3.11 .venv-sheetsage
export VIRTUAL_ENV="$PWD/.venv-sheetsage"
uv pip install "torch==2.8.0" "torchaudio==2.8.0"
uv pip install "transformers==4.45.2" "huggingface-hub==0.36.0" "safetensors" \
               "numpy<2" "scipy" "mir_eval" "pretty_midi" "mido" "setuptools"
.venv-sheetsage/bin/hf download m-a-p/SheetSage2 --local-dir sheetsage2
# ./start then converts the audio encoder to MLX (mlx_sheetsage.py).
echo "SHEETSAGE_SETUP_OK"
