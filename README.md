# NOS-ASR-API

Dockerized HTTP API for Galician Automatic Speech Recognition.

Wraps the [`proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm`](https://huggingface.co/proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm) model from [Proxecto Nós] (https://nos.gal/gl/proxecto-nos) with a small Flask + Gunicorn server.

## Setup

Start by cloning this repository and creating your `models` directory:
```
git clone <this-repo-url> NOS-ASR-API
cd NOS-ASR-API
mkdir -p models
```

Models are pulled automatically from Hugging Face the first time the server starts. The download is cached under `models/<model_id>/`, so the second boot is offline. You can find the Proxecto Nós ASR models on Hugging Face: <https://huggingface.co/proxectonos>.

If you prefer to pre-download the weights instead of waiting for the cold start, run:

```
python -c "
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
repo = 'proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm'
Wav2Vec2ProcessorWithLM.from_pretrained(repo, cache_dir='models/nos_asr_gl')
Wav2Vec2ForCTC.from_pretrained(repo, cache_dir='models/nos_asr_gl')
"
```

After the first run, the layout looks like this:

```
NOS-ASR-API/
├── models/
│   └── nos_asr_gl/
│       └── ... (Hugging Face snapshot: model weights, processor, KenLM .bin)
├── config.json
└── ... (other project files)
```

Once your environment is in place, you must define the models to serve in the `config.json` file (located in the project's root). This file instructs the API on which models to load and what settings to use for each.

### Available models

Proxecto Nós currently publishes two Galician ASR checkpoints on Hugging Face:

| `model_id` (suggested)   | `hf_repo`                                                          | `model_type`     | Notes                                                                                          |
|--------------------------|--------------------------------------------------------------------|------------------|------------------------------------------------------------------------------------------------|
| `nos_asr_gl`             | `proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm`            | `wav2vec2_lm`    | Wav2Vec2 XLSR-53 (large) + bundled KenLM 4-gram (`wiki-gl.arpa.bin`). Heavier (~1.5 GB on disk). |
| `nos_asr_gl_300m`        | `proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl`                       | `wav2vec2`       | Wav2Vec2 XLS-R 300m, greedy CTC (no LM). Lighter, trained on Common Voice 17, FalAI, FLEURS, Parlaspeech-GL, OpenSLR. |

Pick `wav2vec2_lm` when you want LM-rescored output (more fluent Galician) and have RAM and disk budget. Pick `wav2vec2` when you want a smaller, faster model and are OK with raw CTC output.

### `config.json` examples

**Single model — XLSR-53 with KenLM language model** (the default this repo ships with):

```
{
    "languages": {"gl": "Galician"},
    "models": [
        {
            "model_id": "nos_asr_gl",
            "lang": "gl",
            "model_type": "wav2vec2_lm",
            "hf_repo": "proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm",
            "sampling_rate": 16000,
            "load": true,
            "default": true
        }
    ]
}
```

**Single model — XLS-R 300m without LM** (lighter, greedy CTC):

```
{
    "languages": {"gl": "Galician"},
    "models": [
        {
            "model_id": "nos_asr_gl_300m",
            "lang": "gl",
            "model_type": "wav2vec2",
            "hf_repo": "proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl",
            "sampling_rate": 16000,
            "load": true,
            "default": true
        }
    ]
}
```

**Both models loaded side by side** (pick at request time via the `model_id` query parameter; the `default: true` flag wins for plain `?lang=gl` calls):

```
{
    "languages": {"gl": "Galician"},
    "models": [
        {
            "model_id": "nos_asr_gl",
            "lang": "gl",
            "model_type": "wav2vec2_lm",
            "hf_repo": "proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm",
            "sampling_rate": 16000,
            "load": true,
            "default": true
        },
        {
            "model_id": "nos_asr_gl_300m",
            "lang": "gl",
            "model_type": "wav2vec2",
            "hf_repo": "proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl",
            "sampling_rate": 16000,
            "load": true
        }
    ]
}
```

> Loading both models at once roughly doubles the memory footprint of the worker. On a CPU-only host with <6 GB RAM, prefer toggling `load` per deploy instead.

### Configuration Fields

* `languages`: A dictionary mapping language codes (e.g., `"gl"`) to their full names (e.g., `"Galician"`).

* `models`: A list where each object defines an ASR model to be loaded.

    + `model_id`: The public-facing identifier for this model (e.g., `"nos_asr_gl"`). Used as the `model_id` parameter in API calls and as the cache subdirectory under `models/`.
    + `lang`: The language code for this model. It must match a key in the `languages` dictionary.
    + `model_type`: The internal identifier for the ASR system. Use `"wav2vec2_lm"` for a Wav2Vec2 checkpoint with a KenLM decoder, or `"wav2vec2"` for greedy CTC decoding without LM.
    + `hf_repo`: The Hugging Face repository the weights and processor are pulled from (e.g., `"proxectonos/Nos_ASR-..."`).
    + `sampling_rate`: The sampling rate (Hz) expected by the model. Incoming audio is automatically resampled to this rate.
    + `load`: Set to `true` to load this model when the API server starts.
    + `default` *(optional)*: Set to `true` to mark this model as the default for its language. If omitted, the first loaded model for each language becomes the default.

### Paths

Models are identified by their Hugging Face repository (`hf_repo`) rather than by file paths. The processor and weights are cached locally under `models/<model_id>/` (or whatever directory `MODELS_ROOT` points to). If you ship the cache alongside the project, the server will reuse it instead of downloading again.

## Installation

Once you have completed the setup, you can run the server using either Docker (recommended) or a local Python environment.

### Run with docker compose (recommended)

This is the simplest and recommended method. It automatically builds the container, handles all dependencies (including the `kenlm` build toolchain and `ffmpeg`), and sets up the environment for you.

This will take care of all installations for you.

```
# 1. Build the Docker image
# (Only needed the first time or when you change the configuration)
docker compose build

# 2. Start the server
docker compose up

# (Optional) To run the server in the background (detached mode):
docker compose up -d

# To stop the server:
docker compose down
```

### Run with local installation

This method is for development or if you prefer not to use Docker.

Native installs need a working C++ toolchain to build the `kenlm` wheel (`cmake`, a C++ compiler, Boost headers, Eigen, zlib) and `ffmpeg` on `PATH` for audio decoding. The Docker image already provides all of these.

```
# macOS
brew install cmake boost eigen ffmpeg

# Debian/Ubuntu
sudo apt-get install build-essential cmake libboost-all-dev libeigen3-dev ffmpeg libsndfile1
```

> **macOS (Apple Silicon):** if the `kenlm` build fails to find Boost, point it at the
> Homebrew prefix before `pip install`: `export BOOST_ROOT="$(brew --prefix boost)"`.

#### Windows

Native Windows builds of `kenlm` are non-trivial because of the Boost dependency. There are two practical routes:

**Option 1 (recommended): WSL2 + Ubuntu.** Install [WSL2](https://learn.microsoft.com/windows/wsl/install), launch an Ubuntu shell, then follow the Debian/Ubuntu instructions above. Everything else (venv, `pip install -r requirements.txt`, `python run.py`) works exactly the same.

**Option 2: native Windows.** Install the dependencies via your preferred package manager. Examples below use [Chocolatey](https://chocolatey.org/) — adapt to [Scoop](https://scoop.sh/) or [winget](https://learn.microsoft.com/windows/package-manager/winget/) if you prefer.

```
:: Install build toolchain + ffmpeg
choco install -y visualstudio2022-workload-vctools cmake ffmpeg

:: Boost via vcpkg (kenlm only needs the headers)
git clone https://github.com/microsoft/vcpkg %USERPROFILE%\vcpkg
%USERPROFILE%\vcpkg\bootstrap-vcpkg.bat
%USERPROFILE%\vcpkg\vcpkg install boost-system boost-thread boost-program-options eigen3 zlib
setx BOOST_ROOT "%USERPROFILE%\vcpkg\installed\x64-windows"
```

After installing the dependencies, **open a fresh terminal** so the new `PATH` and `BOOST_ROOT` take effect, then continue with the Python steps below. Gunicorn does not run on Windows, so the launcher (`run.py`, see step 3) automatically uses [Waitress](https://docs.pylonsproject.org/projects/waitress/) instead — `waitress` is installed from `requirements.txt` on Windows only.

If `pip install` of `kenlm` fails, the error is almost always Boost not being found — re-check `BOOST_ROOT` and the build tools install.

#### 1. (Recommended) Create a Virtual Environment

It is highly recommended to use a Python virtual environment to avoid package conflicts with your other projects.

```
# Create a new virtual environment named 'asr'
python -m venv asr

# Activate the environment
# On macOS/Linux:
source asr/bin/activate
# On Windows:
.\asr\Scripts\activate
```

#### 2. Install Dependencies

Once your environment is active, install the required packages:

```
pip install -r requirements.txt
```

#### 3. Run the Server

You have two ways to run the server:

##### Option A: Cross-platform launcher (recommended)

The `run.py` launcher works the same on Linux, macOS and Windows. It picks the right
server automatically — `gunicorn` (with `--reload`) on Linux/macOS, `waitress` on
Windows — and binds to port `5052` so it does not clash with the Docker container.

```
# Any OS
python run.py

# Or use the convenience wrappers:
./run_local.sh          # Linux / macOS / WSL
run_local.bat           # Windows (PowerShell / cmd)
```

Settings are controlled via environment variables (with defaults):

| Variable         | Default        | Purpose                                    |
|------------------|----------------|--------------------------------------------|
| `PORT`           | `5052`         | Port to bind                               |
| `WEB_THREADS`    | `4`            | Worker threads (use threads for concurrency)|
| `WEB_WORKERS`    | `1`            | Gunicorn workers (Linux/macOS only)        |
| `WEB_TIMEOUT`    | `300`          | Request timeout (s) for long-audio inference|
| `USE_CUDA`       | `0`            | `1` to enable GPU inference                |

Keep `WEB_WORKERS=1`: each worker loads the full model into memory. Scale horizontally
with replicas, not with workers per replica. The 300 s timeout is needed because
long-audio inference on CPU can easily exceed the default 30 s.

##### Option B: Manually with Gunicorn (Linux/macOS)

If you want direct control over the settings:

```
# Run the server on port 5051
gunicorn server:app -b :5051 --workers 1 --threads 4 --timeout 300

# You can change the port to any you like (e.g., :8080)
gunicorn server:app -b :8080 --workers 1 --threads 4 --timeout 300
```

On native Windows, gunicorn is unavailable — use `python run.py` (Option A) or run
`waitress-serve --listen=*:5051 --threads=4 --channel-timeout=300 server:app` directly.

### Using the GPU (Inference)

You can enable GPU acceleration for inference when running locally or with Docker. This requires an NVIDIA GPU and having the NVIDIA Container Toolkit installed for the Docker method.

#### Enabling GPU with Docker

Edit your `docker-compose.yml` file to make two changes:

1. Set the `USE_CUDA` environment variable to `1`.

2. Uncomment the `deploy` block to give the container access to your GPU.

Your file should look like this after editing:

```
...
    environment:
      - USE_CUDA=1

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 # Or "all" to use all available GPUs
              capabilities: [gpu]
...
```
Note: For more details on Docker GPU support, see the [official documentation](https://docs.docker.com/compose/how-tos/gpu-support/).

#### Enabling GPU for a Local Installation

This is controlled by setting the `USE_CUDA` environment variable to `1` before running the server. It works with any launch method:

```
# Linux / macOS — cross-platform launcher
USE_CUDA=1 python run.py

# Windows (PowerShell)
$env:USE_CUDA="1"; python run.py

# Windows (cmd)
set USE_CUDA=1 && python run.py

# Or manually with gunicorn (Linux/macOS)
USE_CUDA=1 gunicorn server:app -b :5051 --workers 1 --threads 4 --timeout 300
```

The helper wrappers `run_local.sh` / `run_local.bat` default `USE_CUDA` to `0`; export it to `1` in your shell before running them to override.

## API usage

The primary API endpoint for transcription is `/api/asr`.

It accepts **POST** requests with a `multipart/form-data` body and the following fields:

* `audio`: The audio file to transcribe. Supported formats: `wav`, `flac`, `ogg`, `mp3`, `m4a`, `webm`. Audio is automatically converted to mono and resampled to the model's sampling rate (16 kHz by default).
* `model_id` *(optional)*: The model identifier to use (e.g., `nos_asr_gl`). Must match a `model_id` defined in your `config.json`.
* `lang` *(optional)*: The language code. If `model_id` is omitted, the API uses the default model for this language.

If neither `model_id` nor `lang` is provided, the request is rejected with `400`.

**Example with `curl`:**

This command transcribes a Galician audio file using the default Galician model and prints the JSON response:

```
curl -X POST -F "audio=@sample.wav" 'http://localhost:5051/api/asr?lang=gl'
```

> **Port:** examples use `5051` (the Docker-compose mapping). If you run locally with
> `python run.py` / the wrappers, the default port is `5052` — adjust the URL accordingly
> (or set `PORT`).

Response:

```
{
  "model_id": "nos_asr_gl",
  "lang": "gl",
  "text": "ola mundo",
  "duration_s": 1.42
}
```

### Auxiliary endpoints

| Method | Path                | Description                                              |
|--------|---------------------|----------------------------------------------------------|
| GET    | `/`                 | Minimal web UI (upload audio, view result)               |
| GET    | `/api/asr/models`   | List loaded models grouped by language, with LM status   |
| GET    | `/api/asr/check`    | Validate a `model_id` / `lang` combination               |

### Environment variables

| Variable           | Default              | Purpose                              |
|--------------------|----------------------|--------------------------------------|
| `ASR_API_CONFIG`   | `config.json`        | Path to model registry               |
| `MODELS_ROOT`      | `models`             | Where Hugging Face weights are cached|
| `USE_CUDA`         | `0`                  | `1` to enable GPU inference          |
| `MAX_AUDIO_MB`     | `50`                 | Upload size cap                      |
| `HF_HOME`          | `models/.hf_cache`   | Hugging Face cache directory         |

## Demo page

Once the server is running, a simple web-based user interface is available in your browser:

- Docker compose: [http://localhost:5051](http://localhost:5051)
- Local (`python run.py` / wrappers): [http://localhost:5052](http://localhost:5052)

This interface allows you to upload an audio file and see the transcription directly from your browser.

**Customization:**

To change the styling of the demo page, edit `static/css/style.css`. To swap the layout, edit `templates/index.html`.

## License

[Apache 2.0.] (https://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements

This work is funded by the Ministerio para la Transformación Digital y de la Función Pública - Funded by EU – NextGenerationEU within the framework of the project Desarrollo de Modelos ALIA. (Esta publicación del proyecto Desarrollo de Modelos ALIA está financiada por el Ministerio para la Transformación Digital y de la Función Pública y por el Plan de Recuperación, Transformación y Resiliencia – Financiado por la Unión Europea – NextGenerationEU).

Thanks also to Dimensiona for the technical development of this API.
