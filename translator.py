import sys
from pathlib import Path
from config import NLLB_LANGUAGE_MAP

def translate_text_lines(
    input_file: Path,
    output_file: Path,
    source_language: str,
    target_language: str,
) -> None:
    source_code = NLLB_LANGUAGE_MAP.get(source_language)
    target_code = NLLB_LANGUAGE_MAP.get(target_language)

    if not source_code:
        print(f"Error: source language '{source_language}' is not mapped for NLLB translation yet.")
        print(f"Currently supported NLLB language codes: {', '.join(sorted(NLLB_LANGUAGE_MAP))}")
        sys.exit(1)

    if not target_code:
        print(f"Error: target language '{target_language}' is not mapped for NLLB translation yet.")
        print(f"Currently supported NLLB language codes: {', '.join(sorted(NLLB_LANGUAGE_MAP))}")
        sys.exit(1)

    print("Loading translation model for non-English target output...")

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("Error: translation dependencies are missing.")
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # To save memory and speed up translation, use float16 if on CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading native NLLB model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=source_code)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=dtype).to(device)
    
    target_lang_id = tokenizer.convert_tokens_to_ids(target_code)

    print("Translation model is ready.")
    print(f"Translating transcript into '{target_language}'...")

    with open(input_file, "r", encoding="utf-8") as source_file:
        source_lines = [line.rstrip("\n") for line in source_file]

    translatable_lines = []
    line_metadata = []
    for line in source_lines:
        if "] " in line:
            # line is like: "[0.00s -> 5.00s] [Speaker X]: the text"
            # We split by the final "] " before the text. 
            parts = line.split("] ")
            
            # parts will look like ["", ""] depending on brackets. Let's do a more robust split for diarization:
            # Reconstruct the prefix completely up until the colon or just split off the last closing bracket space if it's simpler.
            # Best way: find the first occurrence of actual text.
            # Usually format is "[start -> end] [Speaker X]: text" or "[start -> end] text"
            
            # Let's split from the right by ": " if "Speaker" is in it, else by "] "
            if "[Speaker" in line and "]: " in line:
                idx = line.find("]: ") + 3
                metadata = line[:idx]
                text = line[idx:]
            else:
                idx = line.find("] ") + 2
                metadata = line[:idx]
                text = line[idx:]
                
            # Check if there is an emotion tag like [Happy] at the start of the text
            if text.startswith("[") and "] " in text:
                emotion_end_idx = text.find("] ") + 2
                metadata += text[:emotion_end_idx]
                text = text[emotion_end_idx:]
                
            line_metadata.append(metadata)
            translatable_lines.append(text)
        else:
            line_metadata.append("")
            translatable_lines.append(line)

    total_lines = len(translatable_lines)
    with open(output_file, "w", encoding="utf-8") as translated_file:
        for index, (prefix, text) in enumerate(zip(line_metadata, translatable_lines), start=1):
            if text.strip():
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs, forced_bos_token_id=target_lang_id, max_length=512
                    )
                translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            else:
                translated_text = ""

            translated_file.write(f"{prefix}{translated_text}\n")
            translated_file.flush()

            percent = (index / total_lines) * 100 if total_lines else 100.0
            print(f"Translated {index}/{total_lines} lines ({percent:.2f}%)", flush=True)

    print(f"Translated transcript saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python translator.py <input_file> <output_file> <source_lang> <target_lang>")
        print("Example: python translator.py sample.txt sample.hi.txt zh hi")
        sys.exit(1)
        
    input_f = Path(sys.argv[1])
    output_f = Path(sys.argv[2])
    src_l = sys.argv[3]
    tgt_l = sys.argv[4]
    
    if not input_f.exists():
        print(f"File not found: {input_f}")
        sys.exit(1)
        
    translate_text_lines(input_f, output_f, src_l, tgt_l)
