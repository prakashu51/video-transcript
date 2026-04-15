import os
import sys

NLLB_LANGUAGE_MAP = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
}

def resolve_device_and_compute_type(device_preference: str) -> tuple[str, str, str]:
    preference = device_preference.lower()

    if preference not in {"auto", "cpu", "cuda"}:
        print(f"Error: unsupported device option: {device_preference}")
        print("Use one of: auto, cpu, cuda")
        sys.exit(1)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cuda_blocked = cuda_visible == "-1"

    if preference == "cpu":
        return "cpu", "int8", "CPU forced by user"

    if preference == "cuda":
        return "cuda", "float16", "CUDA requested by user"

    if cuda_blocked:
        return "cpu", "int8", "CPU selected because CUDA_VISIBLE_DEVICES disables GPU"

    return "cuda", "float16", "Auto mode: trying NVIDIA CUDA GPU first"
