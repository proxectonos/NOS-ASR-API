import json
import os
import torch
from transformers import (
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_cache_dir(models_root: str, model_id: str) -> str:
    target = os.path.join(models_root, model_id)
    os.makedirs(target, exist_ok=True)
    return target


def load_models(config_data: dict, models_root: str, use_cuda: bool):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    loaded = {}
    defaults_by_lang = {}

    for entry in config_data.get("models", []):
        if not entry.get("load", False):
            continue

        model_id = entry["model_id"]
        lang = entry["lang"]
        hf_repo = entry["hf_repo"]
        sr = entry.get("sampling_rate", 16000)
        model_type = entry.get("model_type", "wav2vec2_lm")
        cache_dir = _resolve_cache_dir(models_root, model_id)

        print(f"[loader] loading {model_id} from {hf_repo} -> {cache_dir}")

        if model_type == "wav2vec2_lm":
            try:
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(hf_repo, cache_dir=cache_dir)
                has_lm = True
            except Exception as e:
                print(f"[loader] LM processor failed ({e}); falling back to plain processor")
                processor = Wav2Vec2Processor.from_pretrained(hf_repo, cache_dir=cache_dir)
                has_lm = False
        else:
            processor = Wav2Vec2Processor.from_pretrained(hf_repo, cache_dir=cache_dir)
            has_lm = False

        model = Wav2Vec2ForCTC.from_pretrained(hf_repo, cache_dir=cache_dir).to(device).eval()

        loaded[model_id] = {
            "processor": processor,
            "model": model,
            "device": device,
            "lang": lang,
            "language": config_data.get("languages", {}).get(lang, lang),
            "sampling_rate": sr,
            "has_lm": has_lm,
            "hf_repo": hf_repo,
        }

        if entry.get("default") or lang not in defaults_by_lang:
            defaults_by_lang[lang] = model_id

    return loaded, defaults_by_lang


def transcribe(loaded_model: dict, audio: "np.ndarray") -> str:
    processor = loaded_model["processor"]
    model = loaded_model["model"]
    device = loaded_model["device"]
    sr = loaded_model["sampling_rate"]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    if loaded_model["has_lm"]:
        result = processor.batch_decode(logits.cpu().numpy())
        text = result.text[0] if hasattr(result, "text") else result["text"][0]
    else:
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return text.strip()
