# Changelog

All notable changes to NOS-ASR-API. Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Initial project scaffolding.
- Dockerfile (Python 3.10 slim + ffmpeg + KenLM build deps).
- `docker-compose.yml` exposing port `5051:8000`.
- Flask server with endpoints `GET /`, `GET /api/asr/models`, `GET /api/asr/check`, `POST /api/asr`.
- `utils/audio.py` audio loader (soundfile + librosa fallback, mono @ 16 kHz).
- `utils/model_loader.py` loading `Wav2Vec2ForCTC` + `Wav2Vec2ProcessorWithLM` from HuggingFace.
- `config.json` registering `proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm`.
- Minimal HTML UI under `templates/index.html` + `static/css/style.css`.    
- `run_local.sh` for non-Docker development on port `5052`.
- Project docs: `README.md`, `CLAUDE.md`, `TASKS.md`, `docs/`.
- `.gitignore` covering venv, caches, HF model weights, local audio fixtures.

### Changed
- Moved test/demo audio out of `docs/pruebas/` into a top-level `samples/` directory (`samples/*.wav` for committed samples, `samples/audio/` for gitignored downloads). Updated all references in `README.md`, `docs/testing.md`, `docs/architecture.md`, `TASKS.md`, and `.gitignore`.

[Unreleased]: about:blank
