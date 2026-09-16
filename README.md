# YuE2 Studio

[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) 음악 생성 모델을 Apple Silicon Mac 에서
돌리는 로컬 웹 대시보드. 곡 생성, 기존 곡 리믹스, 마스터링, 라이브러리 관리까지 한 곳에서.

모델 카드는 Linux + NVIDIA 24GB VRAM 을 요구한다고 적고 있지만, 배포된 `yue2_infer` 휠에는
MPS 경로가 들어 있다. 이 저장소는 그 위에 대시보드와 주변 도구를 얹은 것이다.

```bash
git clone <저장소> && cd yue2 && ./start
```

## 요구사항

- Apple Silicon Mac (M1 이상), 메모리 16GB 이상 권장
- [uv](https://docs.astral.sh/uv/), ffmpeg
- 디스크 8GB (전사 기능까지 쓰면 11GB)

## 시작하기

```bash
./start
```

가상환경 생성 → 의존성 설치 → 가중치 확인 → 서버 기동 → 브라우저 열기를 순서대로 하고,
이미 끝난 단계는 건너뛴다. 평소 실행에도 같은 명령을 쓴다.

첫 실행은 가중치 7.8GB 를 받는다 (`YuE2-3B` 7.26GB + `YuE2-Vae` 530MB, HF 캐시에 저장).

| 옵션 | |
|---|---|
| `--port <번호>` | 기본 8710 |
| `--no-open` | 브라우저를 열지 않음 |
| `--skip-models` | 가중치 확인 생략 |
| `--with-transcription` | 오디오 전사용 SheetSage2 설치 (2.8GB) |

수동 설치:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
.venv/bin/python download_models.py
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8710
```

`requirements.txt` 는 YuE2 휠을 Hugging Face URL 로 고정한다. PyPI 에 없는 패키지이고,
휠이 `torch` / `transformers` 핀을 함께 들고 온다.

환경 점검은 `.venv/bin/yue2 doctor` — `mps_available: true` 를 확인한다.

## 성능

M5 / 32GB / MPS, 32초 클립 기준 실측.

| 단계 | 소요 | 처리량 |
|---|---|---|
| 악보 계획 (ABC) | 45.5s | 908 토큰, 19.9 tok/s |
| semantic 생성 | 46.9s | 800 토큰, 17.1 tok/s |
| NAR 합성 (ODE 32) | 52.8s | |
| VAE 디코드 | 6.3s | 48kHz 스테레오 |
| **전체** | **171.3s** | 오디오 32초 = 실시간의 5.4배 |

CUDA 그래프 없이 eager 디코딩이므로(`graph_fallback_reason: non_cuda_device`) 공식
벤치마크(RTX 4090 에서 3.6분 곡 71초)와는 규모가 다르다. semantic 토큰 25개가 오디오 1초다.

대략 3분 30초 곡이 13–15분, 최대 길이(9000 토큰)가 22–25분.

## 대시보드

`./start` 가 http://localhost:8710 을 연다.

- **만들기** — 스타일, 가사, 악보 계획(cot), 시드, 길이, 음질, 마스터링 프리셋
- **리믹스** — 오디오를 올리면 전사 후 이어서 생성, 또는 ABC 를 직접 입력
- **진행률** — 단계별 실시간 표시, 토큰/초와 남은 시간
- **라이브러리** — 검색, 제목·스타일·가사 수정, 메모, 즐겨찾기, 원본/마스터링본 전환 재생,
  악보 보기, 마스터링 리포트, 프리셋 바꿔 다시 마스터링
- 각 항목의 `?` 에 설명과 주의사항

작업은 큐에서 하나씩 실행된다. 모델이 동시 처리를 지원하지 않는다.

### 취소와 삭제

| 상태 | 취소 | 삭제 |
|---|---|---|
| `queued` | ○ | ○ |
| `running` | ○ | ✕ — 산출물을 쓰는 중이라 취소 먼저 (API 는 409) |
| `complete` / `failed` / `cancelled` / `interrupted` | — | ○ |

## 드래프트 → 고음질

곡을 정하는 단계와 음질을 정하는 단계가 다르다. ABC 악보와 semantic 토큰이 멜로디·보컬·편곡을
결정하고, NAR/VAE 는 그것을 오디오로 렌더링할 뿐이다. `ode_steps` 가 음질 손잡이다.

같은 토큰을 스텝 수만 바꿔 렌더링한 결과 (32초 분량):

| ODE 스텝 | NAR | 32스텝 렌더와 파형 상관도 |
|---|---|---|
| 4 | 7.4s | 0.993 |
| 8 | 14.1s | 0.999 |
| 16 | 26.2s | 1.000 |
| 32 | 51.9s | 기준 |

ODE 4 는 합성이 7배 빠른데 파형 상관도가 0.993 이다. 초안으로 곡을 확인하고, 마음에 들면
저장된 토큰을 재사용해 다시 렌더링하면 된다. 이 재합성은 비트 단위로 재현된다.

곡이 길이 한도에 걸려 잘렸다면 버튼이 `이어서 풀버전으로` 가 된다. 기존 토큰을 프리픽스에
붙여 디코딩을 이어가므로 앞부분은 같은 음악이고 뒤는 새로 작곡된다. 300 토큰 드래프트를
1200 토큰으로 늘린 실측에서 앞 300 토큰이 그대로 유지됐다. 다만 NAR 이 늘어난 시퀀스
전체를 보고 렌더링하므로 파형까지 같지는 않다(겹치는 구간 상관도 0.85, 엔벨로프 0.98).

### 시드로는 곡을 되살릴 수 없다

| 조건 | semantic 토큰 일치율 |
|---|---|
| 같은 프로세스에서 두 번 생성 | 250/250 (100%) |
| 프로세스를 새로 띄워 같은 시드·악보·설정 | 4/800 (0.5%) |
| 저장된 토큰 재합성 | 비트 단위 동일 |

샘플링 RNG 는 CPU 에 고정 시드로 만들어지므로 난수열은 같다. 갈리는 쪽은 로짓이다.
MPS 가 프로세스마다 다른 커널을 고르면서 bf16 어텐션 결과가 미세하게 달라지고, 그 차이가
두 토큰 만에 샘플링을 갈라놓는다.

**마음에 든 테이크를 지키는 방법은 저장된 산출물뿐이다.** `semantic.npy` + `score.abc` +
`plan.json` 이 곡 자체이고, 서버를 재시작해도 그 토큰으로 재합성하면 같은 연주가 나온다.

같은 원인이 디코딩 중 NaN 으로 나타나기도 한다. `sampling_guard.py` 가 NaN·+inf 로짓을
선택 불가(-inf)로 바꿔 넘긴다. 후보가 전부 사라진 경우에만 멈춘다.

## 리믹스

모델 카드의 [Cover an existing song](https://huggingface.co/m-a-p/YuE2-3B#cover-an-existing-song)
절차를 따른다. 악보를 주면 작곡 단계를 건너뛰고 그 선율 위에 새 편곡과 보컬을 올리므로,
리믹스가 새 곡보다 빠르다 — 32초 클립 기준 54초 대 145초.

리믹스 탭에서 오디오 파일을 고르고 제출하면 전사가 먼저 큐에 들어가고, 악보가 나오면
서버가 이어서 리믹스를 큐에 넣는다. 악보만 필요하면 `전사 대기열에 추가` 를 따로 누른다.

`cot="melody"` 는 화음 기호를 자동으로 지우지 않는다. 화음이 있는 악보를 melody 모드로
넘기면 요청이 거부되고, 선택지는 두 가지다.

| 원하는 것 | 악보 | cot |
|---|---|---|
| 선율만 따라가고 화성은 새로 (권장) | 화음 기호 없음 | `melody` |
| 원곡 화성까지 유지 | 화음 기호 포함 | `full` |

대시보드는 악보를 분석해 자동으로 맞춘다. `score.py` 의 화음 제거는 본문의 `"Dm7"` 같은
따옴표 문자열만 지우고 `V: Vocal name="..."` 같은 헤더 필드는 건드리지 않는다.

### 오디오 전사 (SheetSage2)

```bash
./setup_sheetsage.sh
```

SheetSage2 는 `torch==2.8` / `transformers==4.45` / `numpy<2` 를 요구해 yue2 의 핀과
공존할 수 없다. `.venv-sheetsage` 라는 별도 환경에 설치하고 서브프로세스로 호출하며,
경계를 넘는 것은 ABC 텍스트뿐이다 (`transcribe.py`).

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
| `GET` | `/api/state` | 디바이스, 실행 중·대기 중인 작업 |
| `GET` | `/api/library?search=&status=&favorite=` | 라이브러리 검색 |
| `POST` | `/api/jobs` | 생성 요청 |
| `PATCH` | `/api/jobs/{id}` | 제목·스타일·가사·메모·즐겨찾기 수정 |
| `DELETE` | `/api/jobs/{id}` | 항목과 산출물 삭제 |
| `POST` | `/api/jobs/{id}/cancel` | 취소 |
| `POST` | `/api/jobs/{id}/retry` | 같은 요청 재제출 |
| `POST` | `/api/jobs/{id}/upgrade?ode_steps=32&extend_to=` | 재합성, 선택적으로 연장 |
| `POST` | `/api/jobs/{id}/master?preset=` | 다시 마스터링 |
| `GET` | `/api/jobs/{id}/audio?variant=raw\|mastered` | 오디오 |
| `GET` | `/api/jobs/{id}/abc` | 악보 |
| `GET` | `/api/jobs/{id}/mastering` | 마스터링 리포트 |
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
transcribe.py         SheetSage2 래퍼
sampling_guard.py     MPS NaN 로짓 대응
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
