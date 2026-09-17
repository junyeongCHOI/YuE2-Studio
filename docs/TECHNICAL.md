# YuE2 Studio 기술 문서

[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) 음악 생성 모델을 Apple Silicon Mac 에서
돌리는 로컬 웹 대시보드의 설치·구조·측정 기록. 사용법은 [README](../README.md) 에 있다.

모델 카드는 Linux + NVIDIA 24GB VRAM 을 요구한다고 적고 있지만, 배포된 `yue2_infer` 휠에는
MPS 경로가 들어 있다. 이 저장소는 그 위에 MLX 백엔드, 대시보드, 주변 도구를 얹은 것이다.

- [설치와 실행](#설치와-실행)
- [백엔드: MLX 8비트](#백엔드-mlx-8비트)
- [성능](#성능)
- [정확도](#정확도)
- [재현성: 시드와 저장된 토큰](#재현성-시드와-저장된-토큰)
- [드래프트 → 고음질](#드래프트--고음질)
- [조건의 세기: 악보 강도와 스타일 강도](#조건의-세기-악보-강도와-스타일-강도)
- [리믹스와 오디오 전사](#리믹스와-오디오-전사)
- [마스터링](#마스터링)
- [대시보드 동작](#대시보드-동작)
- [CLI](#cli)
- [API](#api)
- [구조](#구조)
- [라이선스](#라이선스)

## 설치와 실행

### 요구사항

- Apple Silicon Mac (M1 이상), 메모리 16GB 이상 권장
- [uv](https://docs.astral.sh/uv/), ffmpeg
- 디스크 13GB (가중치 7.8GB + MLX 변환본 4.9GB, 전사 기능까지 쓰면 +2.8GB, 그 인코더 MLX 변환본 +1.2GB).
  변환이 끝나면 원본을 지워 5.5GB 로 줄일 수 있다([원본 지우기](#원본-지우기)).
  `--torch` 로만 쓸 거라면 변환본이 필요 없어 8GB

### `./start`

가상환경 생성 → 의존성 설치 → 가중치 확인 → 서버 기동 → 브라우저 열기를 순서대로 하고,
이미 끝난 단계는 건너뛴다. 평소 실행에도 같은 명령을 쓴다.

첫 실행은 가중치 7.8GB 를 받고 (`YuE2-3B` 7.26GB + `YuE2-Vae` 530MB, HF 캐시에 저장),
그것을 MLX 로 변환해 `models/YuE2-3B-mlx-8bit` 에 4.9GB 를 더 쓴다. 변환은 20초쯤
걸리고 한 번만 한다. 변환이 끝나면 원본 YuE2-3B 를 지운다([원본 지우기](#원본-지우기)).

| 옵션 | |
|---|---|
| `--port <번호>` | 기본 8710 |
| `--no-open` | 브라우저를 열지 않음 |
| `--skip-models` | 가중치 확인 생략 |
| `--with-transcription` | 오디오 전사용 SheetSage2 설치 (2.8GB) |
| `--backend <mlx\|torch>` | 곡 생성 백엔드 (기본 `mlx`) |
| `--torch` | `--backend torch` 와 같음 — 릴리스 가중치 그대로 |
| `--model <경로>` | 쓸 MLX 변환본 (기본 `models/YuE2-3B-mlx-8bit`) |
| `--list-models` | 쓸 수 있는 모델을 보여주고 종료 |
| `--keep-originals` | 변환이 끝나도 원본(YuE2-3B, MERT-v2)을 지우지 않음 |

```
$ ./start --list-models
  torch                    PyTorch bf16 (원본)      릴리스 가중치 그대로, MPS
  models/YuE2-3B-mlx-8bit  MLX 8비트 (AR+NAR)       토큰 생성 + 합성 · 4.9GB
```

모델은 대시보드 헤더의 `모델` 드롭다운에서도 바꾼다. 대기열이 비어 있을 때만 바뀌고,
바꾸면 이전 백엔드가 들고 있던 가중치는 메모리에서 내려간다.
`YUE2_BACKEND=torch` 또는 `YUE2_MODEL=<경로>` 환경변수로도 시작 백엔드를 정할 수 있다.

### 수동 설치

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
.venv/bin/python download_models.py
.venv/bin/python mlx_convert.py                    # 기본 백엔드가 쓰는 변환본
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8710
```

`requirements.txt` 는 YuE2 휠을 Hugging Face URL 로 고정한다. PyPI 에 없는 패키지이고,
휠이 `torch` / `transformers` 핀을 함께 들고 온다.

환경 점검은 `.venv/bin/yue2 doctor` — `mps_available: true` 를 확인한다.

### 원본 지우기

변환본은 원본을 완전히 대신한다. 그래서 `./start` 는 변환에 성공한 바로 그 실행에서, 변환이 대신하는
원본을 HF 캐시에서 지운다(`prune_models.py`). 지우지 않으려면 `--keep-originals`.

| 원본 | 크기 | 대신하는 것 | 언제 지우나 |
|---|---|---|---|
| `m-a-p/YuE2-3B` | 6.8GB | `models/YuE2-3B-mlx-8bit` | 곡 생성 모델 변환 직후. 변환본이 토큰 생성·합성·토크나이저를 모두 가질 때만(`mlx_backend.self_contained()`) |
| `m-a-p/MERT-v2-FullSong` | 2.4GB | `models/SheetSage2-encoder-mlx` | 전사 인코더 변환 직후 |

`YuE2-Vae`(오디오 디코드)와 `sheetsage2/`(전사 디코더와 어댑터)는 계속 쓰므로 남는다. 이미 변환을
끝낸 설치는 지우지 않는다. 직접 지우려면:

```bash
.venv/bin/python prune_models.py m-a-p/YuE2-3B m-a-p/MERT-v2-FullSong
```

지운 뒤에는:

- `./start` 가 원본을 다시 받지 않고 VAE 만 확인한다.
- 헤더의 `모델` 목록에서 `PyTorch bf16` 과 AR 전용 변환본이 빠진다. 둘 다 원본 가중치로
  돌기 때문에, 고르면 다음 작업이 7.26GB 를 받게 된다.
- 전사는 MERT-v2 를 읽지 않는다. SheetSage2 는 불러올 때 부모 모델의 코드 두 파일과 가중치를
  해시로 확인하는데, 코드 파일은 `sheetsage2/` 안에 같은 바이트로 들어 있어 그것을 해시하고,
  가중치는 `mlx_sheetsage.py` 가 변환할 때 기록한 원본 해시로 대신한다(`sheetsage_runner.py`).
  MERT-v2 가 없는 빈 캐시에서 인터넷을 막고 253초 곡을 전사해 같은 악보를 얻었다.
- 원본이 필요한 것은 다시 받는다: `./start --torch`·`download_models.py`(YuE2-3B), MLX 인코더 없이
  PyTorch 로만 하는 전사(MERT-v2). `mlx_convert.py`·`mlx_sheetsage.py convert`·`mlx_verify*.py` 도
  원본을 읽는다.

## 백엔드: MLX 8비트

곡 하나에 드는 시간은 토큰을 하나씩 뽑는 단계(AR)와 그 토큰을 오디오로 펴는 단계(NAR)가
거의 전부다. 둘 다 MLX 로 옮겼고, 이쪽이 기본 백엔드다.

변환본을 직접 만들고 검증하려면:

```bash
.venv/bin/python mlx_convert.py           # 기본값: AR 8비트 그룹 64, NAR BF16
.venv/bin/python mlx_verify.py            # 토큰 생성을 torch MPS / CPU FP32 와 비교
.venv/bin/python mlx_verify_nar.py        # 합성을 torch MPS 와 비교 (잠재·파형)
```

### 무엇을 어떻게 변환했나

YuE2-3B 는 Mixture-of-Transformers 라 레이어마다 완전한 스택이 두 벌 있고, 어텐션
연산만 공유한다.

| 스택 | 쓰는 단계 | 저장 형식 |
|---|---|---|
| AR (`self_attn`/`mlp`) + `embed_tokens` + `lm_head` | 악보 계획, semantic 토큰 | **8비트** affine, 그룹 64 |
| NAR (`nar_*`) + `vae2llm`/`llm2vae`/`time_embedder` | 플로우 매칭 합성 | BF16 |
| VAE 디코더 | 오디오 디코드 | 변환하지 않음 (PyTorch) |

NAR 을 8비트로 만들지 않은 건 측정 결과다. 토큰 생성은 한 번에 토큰 하나라 메모리
대역폭에 묶여 있어서 가중치를 절반으로 줄이면 그만큼 빨라진다. 합성은 잠재 프레임
800여 개를 한 번에 보는 계산이라 대역폭이 병목이 아니다 — NAR 까지 8비트로 만들어도
17.5초가 16.9초, 3% 였다. 대신 같은 솔버에서 BF16 과 파형 상관도 0.9991 만큼 갈린다.
속도를 주지 않는 근사는 넣지 않았다. 파일을 1.2GB 줄이려면 `--quantize-nar`.

샘플링은 다시 구현하지 않았다. 로짓만 MLX 에서 받아 `yue2.sampling.distribution` 과
같은 시드의 CPU multinomial 로 넘긴다 — 금지 토큰 마스킹, 반복 페널티, top-k/p 가 전부
상류 코드 그대로다. 합성도 `yue2.nar.song_chunks` 를 그대로 불러 청크 경계와 시드 노이즈
추첨을 공유하고, 32스텝 midpoint 솔버와 BF16 상태 연산까지 같은 순서로 맞췄다.

MLX 백엔드는 6.8GB BF16 모델을 아예 올리지 않고, 파일도 읽지 않는다. 파이프라인을 변환본
폴더 위에 만들기 때문에(토크나이저 `qwen.tiktoken` 은 변환기가 복사해 둔다) 각 take 의
`result.json` 에 기록되는 `weights.mot` 도 실제로 쓴 변환본의 체크섬이다. 메모리에 올라가는
것은 4.9GB MLX 가중치와 530MB VAE 뿐이다.

## 성능

모두 M5 / 32GB, 같은 요청, 32초 클립.

### MLX 대 torch MPS

| 단계 | torch MPS bf16 | MLX |
|---|---|---|
| 악보 계획 (ABC) | 47.8s · 21.5 tok/s | **19.7s · 46.8 tok/s** |
| semantic 생성 | 47.1s · 17.0 tok/s | **17.9s · 44.6 tok/s** |
| NAR 합성 (ODE 32) | 53.9s | **16.2s** |
| VAE 디코드 | 5.1s | 3.4s |
| **전체** | **156.3s** | **58.1s** |

오디오 32초가 실시간의 5.4배에서 1.8배가 됐다. 긴 곡은 컨텍스트가 길어질수록 디코딩도
합성도 느려지므로 이 비율이 그대로 유지되지는 않는다. 두 실행이 같은 곡을 만들지는
않으므로(ABC 1030 토큰 대 924 토큰) 비교해야 하는 건 tok/s 쪽이다. 합성은 양쪽 다
800프레임이라 그대로 비교된다.

### torch MPS 단독 측정

| 단계 | 소요 | 처리량 |
|---|---|---|
| 악보 계획 (ABC) | 45.5s | 908 토큰, 19.9 tok/s |
| semantic 생성 | 46.9s | 800 토큰, 17.1 tok/s |
| NAR 합성 (ODE 32) | 52.8s | |
| VAE 디코드 | 6.3s | 48kHz 스테레오 |
| **전체** | **171.3s** | 오디오 32초 = 실시간의 5.4배 |

CUDA 그래프 없이 eager 디코딩이므로(`graph_fallback_reason: non_cuda_device`) 공식
벤치마크(RTX 4090 에서 3.6분 곡 71초)와는 규모가 다르다. semantic 토큰 25개가 오디오 1초다.
이 백엔드에서 3분 30초 곡은 대략 13–15분, 최대 길이(9000 토큰)는 22–25분이다.

## 정확도

### 토큰 생성 (`mlx_verify.py`, FP32 CPU 기준)

| 프롬프트 | 백엔드 | 코사인 | 최대 절대오차 | KL | greedy 64토큰 일치 |
|---|---|---|---|---|---|
| ABC (123토큰) | MLX 8bit | 0.999995 | 0.111 | 9.1e-10 | 64/64 |
| ABC (123토큰) | MPS bf16 | 0.999997 | 0.111 | 4.6e-09 | 64/64 |
| semantic (1502토큰) | MLX 8bit | 0.999988 | 0.164 | 2.0e-05 | 16/64 |
| semantic (1502토큰) | MPS bf16 | 0.999985 | 0.318 | 7.3e-05 | 중단 (아래) |

8비트 양자화 오차가 bf16 반올림 오차보다 크지 않다. semantic 프롬프트에서는 세 지표 모두
MLX 쪽이 FP32 에 더 가까웠다.

### 합성 (`mlx_verify_nar.py`, 800프레임, ODE 32)

같은 semantic 토큰과 같은 시드 노이즈에서 출발하므로 이쪽은 완전히 결정적이다. 차이는
전부 수치 차이다.

| 비교 | 잠재 상관도 | 상대 RMS | 파형 상관도 | 엔벨로프 |
|---|---|---|---|---|
| MLX vs torch MPS bf16 | 0.999758 | 2.20% | **0.9993** | 0.9999 |
| MLX 8비트 NAR vs MLX BF16 NAR | 0.999721 | 2.36% | 0.9991 | 0.9999 |

파형 상관도 0.9993 은 [드래프트 → 고음질](#드래프트--고음질) 표의 ODE 8 스텝 렌더(0.999)보다
오히려 가까운 거리다. 다른 곡에서 같은 semantic 토큰으로 확인했을 때도 0.9987 이었다.

ODE 는 미세한 차이를 증폭하는 계산이라(64번의 속도 평가) 잠재 상대 RMS 2% 대는 비교
대상이 무엇이든 비슷하게 나온다. 위 두 줄의 크기가 비슷한 것도 그 때문이다.

### 검증 중에 나온 MPS 결함

저장된 take 의 semantic 프리픽스(1502토큰)로 ABC 생성 → semantic 생성을 한 프로세스에서
이어 돌리면, torch MPS 쪽 로짓이 semantic step 1 에서 **184704개 전부 NaN** 이 됐다.
3회 반복 모두 재현됐고, `sampling_guard` 도 후보가 전부 사라진 경우라 살리지 못한다.
같은 프리픽스를 단독으로 돌리면 멀쩡하고(12.6–14.5 tok/s), 새로 만든 곡의
프리픽스(1163토큰)에서는 나오지 않았다. MLX 경로에서는 이번 측정 중 한 번도 없었다.

`sampling_guard.py` 는 그보다 가벼운 경우, 디코딩 중 일부 로짓이 NaN·+inf 가 되는 경우를
맡는다. 그 값을 선택 불가(-inf)로 바꿔 넘기고, 후보가 전부 사라진 경우에만 멈춘다.

## 재현성: 시드와 저장된 토큰

같은 요청·시드로 프로세스를 두 번 새로 띄워 토큰을 비교했다.

| 조건 | ABC 토큰 | semantic 토큰 |
|---|---|---|
| MLX, 프로세스 새로 띄움 | **923/923 (100%)** | **800/800 (100%)** |
| torch MPS, 같은 프로세스에서 두 번 | | 250/250 (100%) |
| torch MPS, 프로세스 새로 띄움 | 894까지 같고 분기 (길이 900 대 949) | 4/800 (0.5%) |
| 저장된 토큰 재합성 (양쪽) | | 비트 단위 동일 |

샘플링 RNG 는 CPU 에 고정 시드로 만들어지므로 난수열은 어느 백엔드든 같다. 갈리는 쪽은
로짓이다. MPS 는 프로세스마다 다른 커널을 고르면서 bf16 어텐션 결과가 미세하게 달라지고,
그 차이가 두 토큰 만에 샘플링을 갈라놓는다. MLX 는 프로세스마다 커널이 갈리지 않아
로짓이 같다.

그래서 **MLX 로 만든 곡은 시드·요청·설정만 같으면 다시 나오고**, torch MPS 로 만든 곡은
시드로 되살릴 수 없다. 어느 쪽이든 가장 확실한 보존 수단은 저장된 산출물이다 —
`semantic.npy` + `score.abc` + `plan.json` 이 곡 자체이고, 서버를 재시작해도 그 토큰으로
재합성하면 같은 연주가 나온다.

## 드래프트 → 고음질

곡을 정하는 단계와 음질을 정하는 단계가 다르다. ABC 악보와 semantic 토큰이 멜로디·보컬·편곡을
결정하고, NAR/VAE 는 그것을 오디오로 렌더링할 뿐이다. `ode_steps` 가 음질 손잡이다.

같은 토큰을 스텝 수만 바꿔 렌더링한 결과 (torch MPS, 32초 분량):

| ODE 스텝 | NAR | 32스텝 렌더와 파형 상관도 |
|---|---|---|
| 4 | 7.4s | 0.993 |
| 8 | 14.1s | 0.999 |
| 16 | 26.2s | 1.000 |
| 32 | 51.9s | 기준 |

ODE 4 는 합성이 7배 빠른데 파형 상관도가 0.993 이다. 초안으로 곡을 확인하고, 마음에 들면
저장된 토큰을 재사용해 다시 렌더링하면 된다(대시보드의 `이 곡 그대로 고음질로`,
API 의 `upgrade`). 이 재합성은 비트 단위로 재현된다.

곡이 길이 한도에 걸려 잘렸다면 버튼이 `이어서 풀버전으로` 가 된다. 기존 토큰을 프리픽스에
붙여 디코딩을 이어가므로 앞부분은 같은 음악이고 뒤는 새로 작곡된다. 300 토큰 드래프트를
1200 토큰으로 늘린 실측에서 앞 300 토큰이 그대로 유지됐다. 다만 NAR 이 늘어난 시퀀스
전체를 보고 렌더링하므로 파형까지 같지는 않다(겹치는 구간 상관도 0.85, 엔벨로프 0.98).

## 조건의 세기: 악보 강도와 스타일 강도

리믹스에서 악보를 너무 문자 그대로 따라간다면 강도를 내릴 수 있다. 대시보드의
**악보 강도** (API 는 `score_scale`, 기본 `1.0`) 다. **스타일 강도** 는 상류의
`cfg_scale` 이다.

악보는 가중치가 붙는 조건이 아니라 프롬프트에 그대로 들어간다 — `token_prefixes` 가
지시문·태그·가사 다음에 ABC 를 붙이고 그 뒤부터 음향 토큰을 뽑는다. 그래서 원래는
"조금만 따라가기"라는 게 없다. 상류의 `cfg_scale` 도 이 일을 하지 못한다. 그쪽
네거티브 브랜치에는 **같은 ABC 가 그대로 들어가서**(`cfg_negative:
same_instruction_and_exact_abc`) 악보는 상쇄되고 태그·가사만 증폭되기 때문이다.

악보를 뺀 프롬프트로 한 번 더 디코딩하면 악보의 기여분이 차이로 드러나고, 거기에
계수를 곱할 수 있다 (`guidance.py`).

```
logits = P + (cfg_scale − 1)(P − N_text) + (score_scale − 1)(P − N_score)

P        지시문 + 태그 + 가사 + 악보        (상류가 보내는 것)
N_text   지시문 + 악보                     (상류의 네거티브)
N_score  지시문 + 태그 + 가사 + 빈 악보     (이 저장소가 더한 것)
```

`score_scale=1` 은 지금까지의 동작과 **산술까지 동일**하고, 0 은 악보를 무시하고
가사·스타일만 보며, 1 보다 크면 외삽이다. `N_score` 는 체크포인트가 악보 없이 쓸 때
보내는 빈 ABC 블록(`cot="off"` 가 `[ABC_START, ABC_END]` 만 넣는다)을 그대로 쓴다.
P 와 다른 것이 악보의 내용뿐이도록.

### 얼마나 움직이나

같은 프리픽스에 한 번 디코딩해 두 브랜치의 로짓을 모아 두면, 어떤 강도든 그 둘의
선형결합이라 한 번에 잰다. 전사 악보로 만든 리믹스 120스텝, 기준은 `score_scale=1`
이 실제로 뽑은 토큰열이다.

| score_scale | 기준 토큰 확률 | 분포 KL | 최상위 토큰 유지 |
|---|---|---|---|
| 0.00 | 0.0325 | 0.616 | 62% |
| 0.25 | 0.0351 | 0.303 | 69% |
| 0.50 | 0.0368 | 0.120 | 80% |
| 0.75 | 0.0380 | 0.028 | 92% |
| **1.00** | **0.0386** | **0** | **100%** |
| 1.25 | 0.0388 | 0.032 | 85% |
| 1.50 | 0.0388 | 0.142 | 73% |
| 2.00 | 0.0381 | 0.531 | 50% |

강도를 내릴수록 악보가 실제로 고른 토큰의 확률이 줄고 분포가 멀어진다. 절대값이
작아 보이지만(코덱 어휘가 32768개다) 매 스텝 누적되므로 결과는 다른 곡이 된다.
1 을 넘으면 KL 이 대칭으로 다시 커진다 — 더 충실해지는 게 아니라 외삽이라 과하면
무너진다는 뜻이다.

### 비용

손잡이 하나가 브랜치 하나다. 둘 다 기본값이 아니면 브랜치가 셋이고, 셋을 보조를
맞춰 디코딩한다. 32초 클립 실측 (MLX):

| 브랜치 | 곡 생성 | |
|---|---|---|
| 1 | 43.0 tok/s | 기본 |
| 2 | 24.3 tok/s | 한쪽 손잡이 |
| 3 | 16.0 tok/s | 양쪽 |

대시보드의 예상 소요가 브랜치 수를 반영한다. `cot=off` 는 악보 자체가 없어
(N_score 가 P 와 같아진다) 악보 강도가 잠기고 브랜치도 늘지 않는다.

## 리믹스와 오디오 전사

모델 카드의 [Cover an existing song](https://huggingface.co/m-a-p/YuE2-3B#cover-an-existing-song)
절차를 따른다. 악보를 주면 작곡 단계를 건너뛰고 그 선율 위에 새 편곡과 보컬을 올리므로,
리믹스가 새 곡보다 빠르다 — 32초 클립 기준 54초 대 145초.

리믹스 탭에서 오디오 파일을 고르고 제출하면 전사가 먼저 큐에 들어가고, 악보가 나오면
서버가 이어서 리믹스를 큐에 넣는다. 악보만 필요하면 `전사 대기열에 추가` 를 따로 누른다.

### 화음 기호와 `cot`

`cot="melody"` 는 화음 기호를 자동으로 지우지 않는다. 화음이 있는 악보를 melody 모드로
넘기면 요청이 거부되고(`strip_chords=true` 를 함께 보내면 서버가 지운다), 선택지는 두 가지다.

| 원하는 것 | 악보 | cot |
|---|---|---|
| 선율만 따라가고 화성은 새로 (권장) | 화음 기호 없음 | `melody` |
| 원곡 화성까지 유지 | 화음 기호 포함 | `full` |

대시보드는 악보를 분석해 자동으로 맞춘다. `score.py` 의 화음 제거는 본문의 `"Dm7"` 같은
따옴표 문자열만 지우고 `V: Vocal name="..."` 같은 헤더 필드는 건드리지 않는다.

### SheetSage2

```bash
./setup_sheetsage.sh          # 또는 ./start --with-transcription
```

SheetSage2 는 `torch==2.8` / `transformers==4.45` / `numpy<2` 를 요구해 yue2 의 핀과
공존할 수 없다. `.venv-sheetsage` 라는 별도 환경에 설치하고 서브프로세스로 호출하며,
경계를 넘는 것은 ABC 텍스트뿐이다 (`transcribe.py`). M5 / 32GB 에서 40초 발췌 전사에 약 19초가
걸렸다.

#### 인코더는 MLX, 디코더는 CPU

SheetSage2 는 곡을 300초 창으로 나눠 보고, 짧은 클립도 300초로 채운다. MERT 인코더(Conformer 24층,
6.3억 파라미터)는 초당 25프레임이라 창 하나가 7500프레임이고, 계산과 메모리 대부분이 여기서 나온다.
디코더(5700만 파라미터)는 그 결과를 읽으며 문법에 맞는 토큰만 골라 수천 스텝을 돈다.

Apple Silicon 에서는 인코더를 MLX 로 옮겨 돌린다 (`mlx_sheetsage.py`). 설치 때
(`setup_sheetsage.sh`, 기존 설치는 `./start`) LoRA 를 병합한 인코더를
`models/SheetSage2-encoder-mlx` (1.2GB) 로 한 번 변환한다. `sheetsage_runner.py` 가 infer.py 를
그대로 실행하되, PyTorch 로 MERT 를 올리는 대신 설정과 멜 프런트엔드만 남기고 인코더 출력을 MLX 에서
받는다. 디코더는 CPU FP32 그대로다.

- **정밀도** — 가중치는 FP16 으로 저장하고 계산은 전부 FP32 다. 인코더 전체를 반정밀도로 돌리면
  (BF16·FP16) 곡 전체에서 디코더가 처음 12% 안에 다른 길로 가서 다른 악보가 됐다(토큰 유사도
  0.80–0.83). FP16 저장 + FP32 계산은 PyTorch FP32 와 악보가 같았다 — 토큰 3535개 중 12개만 박자
  시각이 10–30ms 달랐고 ABC 에는 영향이 없었다. 인코더 출력의 코사인 유사도 0.9999994.
- **어텐션** — 7500 × 7500 행렬은 FP32 로 3.35 GiB 라 쿼리를 1024프레임씩 나눠 계산한다. 마스크도
  인과성도 없어서 행끼리 독립이고, 비트 단위로 같았다.
- **멜 프런트엔드** — 창 전체 STFT 를 한 번에 만들면 CPU 할당자가 0.9GB 를 끝까지 쥐고 있어서
  2048프레임씩 나눠 만든다. 결과는 비트 단위로 같았다.
- **메모리** — 서브샘플링을 층마다 평가하고 MLX 버퍼 캐시를 0 으로 둬서, 한 층의 중간값 이상을
  들고 있지 않는다.

`python mlx_sheetsage.py verify <오디오>` 가 한 창에서 PyTorch FP32 인코더와 출력을 비교한다.

M2 / 8GB 에서 같은 녹음으로 잰 결과 (`--melody-only`, 최대 메모리는 프로세스 phys_footprint):

| 경로 | 15초 발췌 | 253초 곡 전체 | 악보 |
|---|---|---|---|
| MPS, FP32 (이전 기본값) | 최대 11GB, 스왑에 빠져 10분 넘게 첫 창을 못 끝냄 | — | — |
| MPS, 어텐션을 1024프레임씩 나눔 | 5.9GB 넘어 스왑 | — | — |
| MPS, BF16 가중치 | 23초, 3.7GB | — | **깨짐** (디코더가 pad 토큰을 냄) |
| MPS, 인코더 BF16 + 디코더 FP32 | 25초, 3.8GB | 5.4GB 넘어 중단 | 발췌는 CPU 와 같음 |
| CPU, FP32 | 41초, 2.5GB | 88초, 2.48GB | 기준 |
| MLX 인코더 (BF16) + CPU 디코더 | 18초, 2.36GB | 67초, 2.40GB | 곡 전체는 다름 |
| MLX 인코더 (FP32) + CPU 디코더 | — | 71초, 3.93GB | 같음 |
| **MLX 인코더 (FP16 저장·FP32 계산) + CPU 디코더** | **17.5초, 2.20GB** | **61초, 2.36GB** | **같음** |

BF16·FP32 인코더 줄은 서브샘플링 층별 평가를 넣기 전에 잰 값이라 최대 메모리가 조금 높게 나왔을 수 있다.
인코더만 떼면 30초 발췌 한 창에 PyTorch CPU 35초, MLX 9.6초다. 곡 전체에서 남는 시간은 대부분
CPU 디코더다.

`transcribe.py` 는 변환본과 `.venv-sheetsage` 안의 mlx 가 있으면 이 경로를 쓰고
(`/api/transcription` 의 `device` 가 `mlx`), 없으면 PyTorch 로 전부 돌린다. 그때 장치는
`hw.memsize` 가 32GB 이상이면 MPS, 아니면 CPU 다 — 16–24GB 는 재 보지 않았다. 대시보드의 전사 예상
시간은 이 `device` 에 맞춰 위 실측을 쓴다.

올린 녹음은 `outputs/uploads/` 에 남아 전사 항목과 수명을 같이한다. 라이브러리에서
그 항목의 플레이어가 재생하는 것이 이 원본이고, 항목을 지우면 원본도 함께 지워진다.
악보가 원곡을 제대로 따라갔는지 들어보고 확인하라고 남겨둔 것이다.

## 마스터링

모델이 아니라 numpy/scipy 신호처리다 (`mastering.py`).

| 단계 | 구현 |
|---|---|
| Spectral denoise | STFT 스펙트럴 게이트. 빈별 저백분위수를 노이즈 플로어로 잡고, 그보다 `sensitivity_db` 이상 높은 빈은 그대로 통과. 마스크는 채널 최댓값에서 한 번만 계산해 양 채널에 동일 적용 |
| Dynamic EQ | 컷 전용 dynamic bell 4개 (280Hz, 550Hz, 3.2kHz, 7.5kHz). 임계값은 각 대역 자신의 75퍼센타일 상대값 |
| Multiband compression | 120 / 800 / 5000Hz Linkwitz-Riley 4차 4밴드. 상보 분할이라 밴드 합이 입력과 일치 (측정 오차 1.1e-16) |
| Stereo correction | M/S 변환 후 베이스 모노화, side/mid RMS 비율 상한, 상관도 -0.5 미만이면 위상 반전 교정 |
| Limiter + LUFS | ITU-R BS.1770-4 적분 러드니스 → 목표까지 정규화 → 룩어헤드 리미터. 4배 오버샘플링 트루피크 |

리미터 게인 곡선은 룩어헤드 최소값 필터 뒤에 같은 폭의 Hann 스무딩이다. 스무딩 결과의 각
샘플은 자기 자신을 포함하는 구간의 최소값들을 가중평균한 값이므로 항상 샘플별 목표 게인
이하이고, 구조적으로 오버슈트할 수 없다.

리미팅은 러드니스를 깎으므로 정규화 게인을 리미팅 결과 기준으로 재계산하며 반복한다.
게인 1dB 가 LUFS 1dB 를 사주지 않으므로 기울기를 측정해 스텝을 조정하고, 기울기가
`saturation_slope` 아래로 떨어지면 탐색을 멈추고 `reached_target: false` 를 리포트에 남긴다.

| 프리셋 | 목표 | 천장 | 실측 (32초 클립) |
|---|---|---|---|
| `streaming` | -14 LUFS | -1.0 dBTP | -14.07, 최대 GR 6.0dB |
| `loud` | -9 LUFS | -1.0 dBTP | -8.99, 최대 GR 6.0dB (소프트 클립) |
| `gentle` | -16 LUFS | -1.5 dBTP | -16.01, 최대 GR 6.0dB |
| `transparent` | -14 LUFS | -1.0 dBTP | 러드니스+리미터만 |

```bash
.venv/bin/python mastering.py input.flac output.flac --preset streaming
.venv/bin/python mastering.py input.flac output.flac --preset loud --target-lufs -10
```

```python
import soundfile as sf, mastering

audio, sr = sf.read("song.flac", always_2d=True, dtype="float64")
processed, report = mastering.master(audio, sr, "streaming")
sf.write("song.mastered.flac", processed, sr, subtype="PCM_24")
```

`MasteringSettings` 를 직접 구성하면 단계별로 끄고 켜거나 임계값·시정수를 조정할 수 있다.
모든 단계를 끄면 입력이 그대로 보존된다 (float32 변환 오차 외 차이 없음).

## 대시보드 동작

- 작업은 큐에서 하나씩 실행된다. 모델이 동시 처리를 지원하지 않는다. 진행률은 단계별로
  토큰/초와 남은 시간을 표시한다.
- 모델 가중치는 곡을 만드는 동안에만 메모리에 있다. 생성 작업이 이어서 대기 중이면 그대로
  두고, 대기열이 비거나 다음이 전사면 내린다. 헤더의 `model` 이 `warm` 이면 올라가 있는
  상태다. M2 / 8GB 에서 12초 클립을 만들 때 서버 프로세스가 생성 중 5.1–7.9GB, 끝난 뒤
  0.8GB, 이어진 전사 동안 0.4–0.6GB 였다.
- 전사는 작업마다 `.venv-sheetsage` 서브프로세스로 돌고, 끝나거나 취소되면 프로세스와 함께
  SheetSage2·MERT 가 메모리에서 사라진다. YuE2 가중치는 전사 전에 내려가므로 두 모델이 함께
  올라가 있지 않다.
- 왼쪽 폼의 설정은 브라우저(localStorage)에 남아 다음에 열 때 그대로 복원된다 — 스타일,
  가사, 악보, 악보 계획, 시드, 길이, 마스터링, 음질, 두 강도, 탭까지. 제목과 파일 선택은
  한 곡에 속한 것이라 복원하지 않는다. 저장이 막힌 환경(시크릿 창 등)에서는 기본값으로
  동작한다.
- 각 입력의 `?` 에 설명과 주의사항이 있다. 마우스를 올리거나 Tab 으로 이동하면 뜨고,
  Esc 로 닫는다.
- 테마는 헤더 오른쪽 `테마` 에서 시스템·라이트·다크 중에 고른다. 기본은 OS 설정을 따르고,
  고른 값은 브라우저에 남는다. 다크는 Material 다크 테마(#121212 바탕, 높이에 따라 밝아지는
  표면, 채도를 낮춘 강조색)를, 라이트는 같은 계열의 Google 팔레트를 쓴다.
- 색은 WCAG 2.2 AA 에 맞췄다. 두 테마 모두 글자는 배경 대비 4.5:1 이상(1.4.3), 입력칸
  테두리·포커스 링·즐겨찾기 아이콘·진행 막대는 3:1 이상(1.4.11)이다. 페이지에 실제로 그려진
  글자 212곳과 컨트롤 테두리 43곳을 양쪽 테마에서 계산해 확인했다. 색 토큰은
  `static/index.html` 맨 위에 있고, 값을 바꾸면 이 대비를 다시 확인해야 한다.
- 그 밖에: 키보드 포커스 표시, 움직임 줄이기 설정에서 애니메이션 정지, Windows 고대비
  모드에서 선택 상태 유지, 헤더 선택 상자와 즐겨찾기·제목에 스크린리더용 이름.

### 취소와 삭제

| 상태 | 취소 | 삭제 |
|---|---|---|
| `queued` | ○ | ○ |
| `running` | ○ | ✕ — 산출물을 쓰는 중이라 취소 먼저 (API 는 409) |
| `complete` / `failed` / `cancelled` / `interrupted` | — | ○ |

## CLI

대시보드 없이 쓸 수도 있다.

```bash
.venv/bin/yue2 generate --config configs/quick.json --output outputs/quick \
  --style "Funk / nu-disco, warm female vocal, 110 BPM" \
  --lyrics-file examples/lyrics.txt --seed 123

.venv/bin/yue2 generate --abc-file melody.abc --cot melody \
  --style "Jazz-funk, warm lead vocal, Rhodes piano, 96 BPM" \
  --lyrics-file cover_lyrics.txt --output outputs/cover
```

`configs/quick.json` 은 semantic max_tokens 를 800(약 30초)으로 제한한 설정이다.
CLI 에는 화음 검사가 없으므로 멜로디만 쓸 거라면 직접 확인한다.

```bash
.venv/bin/python -c "import score,sys; print(score.analyse(open(sys.argv[1]).read()))" melody.abc
```

## API

| 메서드 | 경로 | |
|---|---|---|
| `GET` | `/api/state` | 디바이스, 현재 모델, 실행 중·대기 중인 작업 |
| `GET` | `/api/models` | 바꿀 수 있는 백엔드·모델 목록 |
| `POST` | `/api/models/select` | 모델 전환 (실행 중이면 409) |
| `GET` | `/api/library?search=&status=&favorite=` | 라이브러리 검색 |
| `POST` | `/api/jobs` | 생성 요청 (`score_scale`, `cfg_scale` 포함) |
| `PATCH` | `/api/jobs/{id}` | 제목·스타일·가사·메모·즐겨찾기 수정 |
| `DELETE` | `/api/jobs/{id}` | 항목과 산출물 삭제 |
| `POST` | `/api/jobs/{id}/cancel` | 취소 |
| `POST` | `/api/jobs/{id}/retry` | 같은 요청 재제출 |
| `POST` | `/api/jobs/{id}/upgrade?ode_steps=32&extend_to=` | 재합성, 선택적으로 연장 |
| `POST` | `/api/jobs/{id}/master?preset=` | 다시 마스터링 |
| `GET` | `/api/jobs/{id}/audio?variant=raw\|mastered\|source` | 오디오 (`source` 는 전사에 쓴 업로드 원본) |
| `GET` | `/api/jobs/{id}/abc` | 악보 |
| `GET` | `/api/jobs/{id}/mastering` | 마스터링 리포트 |
| `GET` | `/api/transcription` | SheetSage2 설치 여부 |
| `POST` | `/api/transcribe` | 오디오 업로드 → 전사 (`remix` 로 생성까지 연결) |
| `POST` | `/api/abc/analyse`, `/api/abc/strip-chords` | 악보 분석·화음 제거 |

서버는 127.0.0.1 에만 바인딩하고 인증이 없다. 브라우저에서 온 교차 출처 요청과 DNS
리바인딩을 막기 위해 Host 헤더가 루프백이 아니면 421 로 거부한다. 신뢰할 수 없는
네트워크에 노출하지 말 것.

## 구조

```
start                 설치부터 실행까지
server.py             FastAPI 앱, 작업 큐, 파이프라인
store.py              SQLite 라이브러리
mastering.py          마스터링 체인
score.py              ABC 악보 분석
guidance.py           악보·스타일 조건의 CFG (브랜치 구성과 결합식)
transcribe.py         SheetSage2 래퍼
sampling_guard.py     MPS NaN 로짓 대응
mlx_convert.py        체크포인트를 MLX 로 변환 (AR 8비트, NAR BF16)
mlx_yue2.py           MoT 레이어의 MLX 구현과 변환본 로더
mlx_ar.py             토큰 생성 루프
mlx_nar.py            플로우 매칭 솔버 (yue2/nar.py 의 포트)
mlx_backend.py        파이프라인의 AR·NAR 단계를 MLX 로 전환
mlx_verify.py         토큰 생성을 torch MPS / CPU FP32 와 비교
mlx_verify_nar.py     합성을 torch MPS 와 비교 (잠재·파형)
mlx_sheetsage.py      SheetSage2 인코더(MERT-v2)의 MLX 구현, 변환, PyTorch 대비 검증
sheetsage_runner.py   SheetSage2 infer.py 를 MLX 인코더 + CPU 디코더로 실행
static/index.html     대시보드
```

산출물은 `outputs/library/<job_id>/` 에, 메타데이터는 `outputs/library.db` 에 저장된다.

## 라이선스

이 저장소의 코드에는 아직 라이선스가 정해져 있지 않다.

받아오는 모델은 모두 **CC BY-NC 4.0**, 비상업적 용도에 한한다 —
[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
[YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae),
[SheetSage2](https://huggingface.co/m-a-p/SheetSage2),
[MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong).
생성한 음원도 같은 제약을 받는다.

리믹스는 원곡의 권리와 별개다. 전사와 리믹스가 원곡에 대한 권리를 만들어 주지는 않는다.
