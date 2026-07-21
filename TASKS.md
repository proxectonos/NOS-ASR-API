# TASKS

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

## Milestone 1 — Scaffolding (2026-05-19)

- [x] Initial project layout (Dockerfile, compose, server, utils, templates)
- [x] Identify HuggingFace model: `proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm`
- [x] Flask endpoints: `/api/asr`, `/api/asr/models`, `/api/asr/check`
- [x] Audio decoding chain (soundfile + librosa + ffmpeg)
- [x] Initial `config.json` with the wav2vec2+LM model
- [x] `.gitignore`, README, CLAUDE.md, CHANGELOG, docs scaffold

## Milestone 2 — First successful transcription

- [x] `docker compose build` succeeds locally on amd64
- [x] Container boots and downloads HF model into `./models/`
- [x] `GET /api/asr/models` returns the registered model
- [x] `POST /api/asr` returns a non-empty transcript for a Galician WAV sample
- [x] Confirm KenLM decoding is active (`has_lm: true` in `/api/asr/models`)
- [x] Add 1–2 short Galician audio samples under `samples/`
- [x] Document curl smoke test in `docs/testing.md`
- [ ] Replace synthetic (es-voice) sample with a native Galician recording for accuracy benchmarking

## Milestone 3 — Hardening

- [ ] Add `/healthz` endpoint (liveness/readiness)
- [ ] Stream-friendly upload (chunked) for long audio
- [ ] Optional VAD / silence trimming before inference
- [ ] GPU path verified (`USE_CUDA=1` with nvidia runtime)
- [ ] Concurrency test: N parallel requests, measure latency / RSS
- [ ] Graceful handling of unsupported codecs (clear 4xx message)

## Milestone 4 — Polish & release

- [ ] Add `LICENSE`
- [ ] CI: lint + Docker build smoke test
- [ ] Publish image tag (`nos-asr-api:0.1.0`)
- [ ] Document deployment recipe (compose + reverse proxy)

