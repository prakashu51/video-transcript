# Video Transcript & Translation Generator

A modular Python pipeline that extracts speaker-attributed, timestamped transcripts from audio/video files and optionally translates them into 12+ languages.

## Features

- **Speaker Diarization** — Identifies and labels individual speakers using [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **Transcription** — Generates accurate timestamped transcripts using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Automatic Language Detection** — Whisper auto-detects the source language
- **English Translation** — High-quality native Whisper translation for any language → English
- **Multi-language Translation** — Translates to 12+ languages via [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) with an English bridge for superior quality
- **Device Flexibility** — Supports `auto`, `cpu`, and `cuda` (NVIDIA GPU)
- **Live Progress** — Real-time progress bars and periodic status updates during long runs

## Architecture

```
extract_audio.py     → Video/audio → WAV extraction (no system ffmpeg needed)
main.py              → CLI entry point & pipeline orchestration
├── diarizer.py      → Speaker diarization (pyannote.audio)
├── transcriber.py   → Whisper transcription & segment alignment
├── translator.py    → NLLB-200 translation (standalone or integrated)
├── audio_utils.py   → Shared audio utilities (duration, output paths)
└── config.py        → Language mappings, device & RAG configuration

chat.py              → Streamlit chat UI ("Chat with Transcript")
└── rag_engine.py    → RAG pipeline: ingest, embed, query (Ollama + ChromaDB)
```

## Translation Strategy: English Bridge

For non-English target languages, the pipeline uses a **two-hop bridge** through English for dramatically better translation quality:

```
Source Audio (e.g. Chinese)
  │
  ├─→ Whisper transcribe  → sample.txt     (native transcript)
  ├─→ Whisper translate   → sample.en.txt  (high-quality English)
  └─→ NLLB (en → target)  → sample.hi.txt  (English → Hindi)
```

Direct translation between distant language pairs (e.g. Chinese → Hindi) produces poor results because training data for those pairs is scarce. Bridging through English leverages Whisper's excellent translation engine and NLLB's strongest language pair (English → X).

## Requirements

1. **Python 3.11+**
2. **Hugging Face Token** — required for pyannote speaker diarization models
3. **Ollama** — required for the Chat with Transcript feature (local LLM)
4. **ffmpeg** *(optional)* — only needed if using the legacy `extract_audio.bat`. The Python-based `extract_audio.py` uses faster-whisper's bundled ffmpeg and requires no system installation.

### Python Dependencies

Install all dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Hugging Face Setup

1. Create a [Hugging Face](https://huggingface.co/) account
2. Generate an access token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the user agreements for these gated models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
```

### Ollama Setup (for Chat with Transcript)

1. Install [Ollama](https://ollama.com) for Windows:

```powershell
irm https://ollama.com/install.ps1 | iex
```

2. Pull the required models:

```powershell
ollama pull llama3.2
ollama pull mxbai-embed-large
```

3. Verify Ollama is running:

```powershell
ollama list
```

## Usage

### 1. Extract audio from video

```powershell
python extract_audio.py sample.mp4
```

Creates `sample.wav` (16kHz mono WAV). No system ffmpeg installation required.

### 2. Transcribe only (with speaker labels)

```powershell
python main.py sample.wav
```

Output: `sample.txt`

```text
[0.00s -> 3.90s] [Speaker 1]: 嗚啊……
[3.90s -> 6.50s] [Speaker 1]: 烤雞 芒果
[6.50s -> 8.12s] [Unknown]: 哇!!!
```

### 3. Transcribe + English translation

```powershell
python main.py sample.wav auto medium auto en
```

Output: `sample.txt` + `sample.en.txt`

### 4. Transcribe + Hindi translation (via English bridge)

```powershell
python main.py sample.wav auto medium auto hi
```

Output: `sample.txt` + `sample.en.txt` + `sample.hi.txt`

### 5. Standalone translation (skip diarization & transcription)

If you already have a transcript file, translate it directly:

```powershell
python translator.py sample.en.txt sample.hi.txt en hi
```

### 6. Chat with Transcript (RAG)

Launch the Streamlit chat interface:

```powershell
python -m streamlit run chat.py
```

This opens a browser-based chat UI at `http://localhost:8501` where you can:
1. Select a transcript file from the sidebar
2. Click **Index** to process it into the vector store
3. Ask natural language questions about the content

**CLI alternative** — you can also use the RAG engine directly:

```powershell
# Index a transcript
python rag_engine.py ingest sample.en.txt

# Query it
python rag_engine.py query sample_en "What was the conversation about?"

# List all indexed transcripts
python rag_engine.py list
```

## Command Format

```text
python main.py <audio_file> [source_language|auto] [model_size] [device] [target_language]
```

| Argument | Default | Options |
|---|---|---|
| `audio_file` | *(required)* | Path to WAV file |
| `source_language` | `auto` | `auto`, `zh`, `en`, `ja`, `hi`, `fr`, `de`, etc. |
| `model_size` | `medium` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `device` | `auto` | `auto`, `cpu`, `cuda` |
| `target_language` | *(none)* | `en`, `hi`, `fr`, `de`, `es`, `ja`, `ko`, etc. |

## Supported Translation Languages

| Code | Language |
|---|---|
| `ar` | Arabic |
| `de` | German |
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `hi` | Hindi |
| `it` | Italian |
| `ja` | Japanese |
| `ko` | Korean |
| `pt` | Portuguese |
| `ru` | Russian |
| `zh` | Chinese (Simplified) |

To add more languages, update the `NLLB_LANGUAGE_MAP` dictionary in `config.py`.

## Pipeline Phases

When you run `python main.py sample.wav auto medium auto hi`, the pipeline executes:

| Phase | Engine | Output |
|---|---|---|
| **Phase 1: Diarization** | pyannote.audio | Speaker timeline segments |
| **Phase 2: Transcription** | faster-whisper | `sample.txt` (native language) |
| **Phase 3a: English Bridge** | faster-whisper (translate) | `sample.en.txt` |
| **Phase 3b: Target Translation** | NLLB-200 | `sample.hi.txt` |

## Device Behavior

| Argument | Behavior |
|---|---|
| `auto` | Try NVIDIA CUDA first, fall back to CPU |
| `cpu` | Force CPU mode |
| `cuda` | Force NVIDIA GPU |

> **Note:** GPU acceleration requires an NVIDIA GPU with CUDA support. On systems without CUDA, `auto` gracefully falls back to CPU.

## Project Files

| File | Purpose |
|---|---|
| `main.py` | CLI entry point, pipeline orchestration |
| `diarizer.py` | Speaker diarization via pyannote.audio |
| `transcriber.py` | Whisper inference & speaker-segment alignment |
| `translator.py` | NLLB translation (also usable standalone) |
| `audio_utils.py` | Audio duration calculation, output path generation |
| `config.py` | NLLB language map, device/compute & RAG configuration |
| `extract_audio.py` | Video → WAV extraction (uses bundled ffmpeg) |
| `extract_audio.bat` | Legacy ffmpeg wrapper (requires system ffmpeg) |
| `rag_engine.py` | RAG pipeline: transcript ingestion, embedding, querying |
| `chat.py` | Streamlit web chat UI for "Chat with Transcript" |
| `requirements.txt` | Python dependencies |
| `.env` | Hugging Face token (not committed to git) |
| `.gitignore` | Git ignore rules for outputs, cache, and secrets |

## Example Workflows

### Chinese audio → Chinese transcript only

```powershell
python extract_audio.py sample.mp4
python main.py sample.wav zh
```

### Chinese audio → English transcript

```powershell
python main.py sample.wav auto medium auto en
```

### Chinese audio → Hindi transcript

```powershell
python main.py sample.wav auto medium auto hi
```

### Auto-detect source → French transcript

```powershell
python main.py sample.wav auto medium auto fr
```

### Quick re-translate existing English transcript to Korean

```powershell
python translator.py sample.en.txt sample.ko.txt en ko
```

### Chat with an English transcript

```powershell
python -m streamlit run chat.py
```

## Common Notes

- First run downloads Whisper, pyannote, and NLLB models (~3-5 GB total)
- Diarization + transcription on CPU takes ~5-10 minutes for a 2.5-minute audio file
- The English bridge translation adds one extra Whisper pass but produces far better results
- CPU-only translation with NLLB is slower but fully functional
- If the detected source language matches the target, no translation is performed
- Chat with Transcript requires Ollama running locally with `llama3.2` and `mxbai-embed-large` models

## Current Limitations

- No `.srt` subtitle export yet
- No batch folder processing
- NLLB coverage limited to 12 mapped languages (easily extensible in `config.py`)

## Future Improvements

- Add `.srt` export
- Add batch processing for multiple videos
- Add more NLLB language mappings
- Translation progress batching for faster throughput
- Multi-language RAG support (chat in native language)
