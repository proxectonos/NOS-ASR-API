#!flask/bin/python
import os

from flask import Flask, render_template, request, jsonify

from utils.audio import load_audio
from utils.model_loader import read_config, load_models, transcribe


MODELS_ROOT = os.getenv("MODELS_ROOT", "models")
CONFIG_JSON_PATH = os.getenv("ASR_API_CONFIG", "config.json")
USE_CUDA = os.getenv("USE_CUDA") == "1"
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "50"))


config_data = read_config(CONFIG_JSON_PATH)
loaded_models, default_model_ids = load_models(config_data, MODELS_ROOT, USE_CUDA)

print("USE_CUDA", USE_CUDA)
print("LOADED MODELS:", list(loaded_models.keys()))
print("DEFAULTS:", default_model_ids)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_MB * 1024 * 1024


@app.route("/")
def index():
    return render_template(
        "index.html",
        models={k: loaded_models[k]["language"] for k in loaded_models.keys()},
    )


@app.route("/api/asr/models", methods=["GET"])
def list_models():
    by_lang = {}
    for mid, m in loaded_models.items():
        lang = m["lang"]
        if lang not in by_lang:
            by_lang[lang] = {"name": m["language"], "models": {}}
        by_lang[lang]["models"][mid] = {
            "default": default_model_ids.get(lang) == mid,
            "sampling_rate": m["sampling_rate"],
            "has_lm": m["has_lm"],
            "hf_repo": m["hf_repo"],
        }
    return by_lang, 200


@app.route("/api/asr/check", methods=["GET"])
def check(model_id=None, lang=None):
    if not model_id and not lang:
        model_id = request.args.get("model_id")
        lang = request.args.get("lang")

    if model_id:
        if model_id not in loaded_models:
            return jsonify({"message": f"Model {model_id} not found"}), 400
        if lang and loaded_models[model_id]["lang"] != lang:
            return jsonify({"message": f"Model {model_id} not in lang {lang}"}), 400
    elif lang:
        if lang in default_model_ids:
            model_id = default_model_ids[lang]
        else:
            return jsonify({"message": f"No model for language {lang}"}), 400
    else:
        return jsonify({"message": "Request must specify model_id or lang"}), 400

    return {
        "model_id": model_id,
        "sampling_rate": loaded_models[model_id]["sampling_rate"],
    }, 200


@app.route("/api/asr", methods=["POST"])
def asr():
    model_id = request.args.get("model_id") or request.form.get("model_id")
    lang = request.args.get("lang") or request.form.get("lang")

    if "audio" not in request.files:
        return jsonify({"message": "Missing 'audio' file field"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"message": "Empty audio file"}), 400

    r, status = check(model_id, lang)
    if status != 200:
        return r, status

    model_id = r["model_id"]
    sr = r["sampling_rate"]

    print(f"ASR API REQUEST: model={model_id} bytes={len(audio_bytes)} filename={audio_file.filename}")

    try:
        audio = load_audio(audio_bytes, target_sr=sr)
    except Exception as e:
        return jsonify({"message": f"Failed to decode audio: {e}"}), 400

    if audio.size == 0:
        return jsonify({"message": "Decoded audio is empty"}), 400

    try:
        text = transcribe(loaded_models[model_id], audio)
    except Exception as e:
        print(f"INFERENCE ERROR: {e}")
        return jsonify({"message": f"Inference failed: {e}"}), 500

    return jsonify({
        "model_id": model_id,
        "lang": loaded_models[model_id]["lang"],
        "text": text,
        "duration_s": float(len(audio)) / float(sr),
    }), 200


def main():
    app.run(debug=True, host="::", port=5002)


if __name__ == "__main__":
    main()
