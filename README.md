# Video Transcript And Translation Generator

This project extracts audio from a video file and generates timestamped transcripts with `faster-whisper`.

It also supports translated transcript output:

- original-language transcript for any Whisper-supported source language
- English translation using Whisper itself
- other target languages using a second translation stage with NLLB

## Current Pipeline

For an input like `sample.mp4`, the repo works like this:

1. [extract_audio.bat](/d:/TG/whisper/extract_audio.bat) extracts audio to `sample.wav`
2. [transcribe.py](/d:/TG/whisper/transcribe.py) runs Whisper transcription
3. if no target language is requested:
   - it writes `sample.txt`
4. if target language is `en`:
   - it writes `sample.txt` for the original transcript
   - it writes `sample.en.txt` using Whisper `task="translate"`
5. if target language is something else like `hi` or `fr`:
   - it writes `sample.txt` for the original transcript
   - it writes `sample.hi.txt` or `sample.fr.txt` using NLLB translation

## Features

- same-name WAV generation from video files
- original transcript generation for any Whisper-supported language
- automatic language detection
- optional device selection: `auto`, `cpu`, `cuda`
- optional English translation through Whisper
- optional non-English translation through NLLB
- live transcript writing while processing
- periodic status logging during long runs
- timestamped transcript lines

## Requirements

You need:

1. Python 3.11 or later
2. `ffmpeg`
3. Python packages from [requirements.txt](/d:/TG/whisper/requirements.txt)

## Python Dependencies

Current dependencies:

- `faster-whisper`
- `torch`
- `transformers`
- `sentencepiece`

Install them with:

```powershell
pip install -r requirements.txt
```

## ffmpeg Requirement

Audio extraction requires `ffmpeg`.

Check with:

```powershell
ffmpeg -version
```

If PowerShell says it is not recognized, install `ffmpeg` and add it to `PATH`.

## Usage

### 1. Extract audio from video

```powershell
.\extract_audio.bat .\sample.mp4
```

This creates:

```text
sample.wav
```

### 2. Generate original transcript only

```powershell
py -X utf8 transcribe.py sample.wav
```

This writes:

```text
sample.txt
```

### 3. Generate English translation

```powershell
py -X utf8 transcribe.py sample.wav auto medium auto en
```

This writes:

```text
sample.txt
sample.en.txt
```

Behavior:

- `sample.txt` is the original-language transcript
- `sample.en.txt` is created using Whisper translation

### 4. Generate another target language

Example for Hindi:

```powershell
py -X utf8 transcribe.py sample.wav auto medium auto hi
```

This writes:

```text
sample.txt
sample.hi.txt
```

Behavior:

- `sample.txt` is the original-language transcript
- `sample.hi.txt` is translated from the original transcript using NLLB

## Command Format

```text
python transcribe.py <audio_file> [source_language|auto] [model_size] [device] [target_language]
```

Arguments:

- `audio_file`: input WAV file
- `source_language|auto`: source language code or `auto`
- `model_size`: Whisper model, default `medium`
- `device`: `auto`, `cpu`, or `cuda`
- `target_language`: optional translated output language

Examples:

```powershell
py -X utf8 transcribe.py sample.wav
py -X utf8 transcribe.py sample.wav auto medium auto
py -X utf8 transcribe.py sample.wav zh medium cpu
py -X utf8 transcribe.py sample.wav zh medium auto en
py -X utf8 transcribe.py sample.wav zh medium auto hi
py -X utf8 transcribe.py sample.wav en small auto fr
```

## Translation Routing Logic

The current repo uses this rule:

- no `target_language`
  - transcription only
- `target_language=en`
  - original transcript with Whisper `task="transcribe"`
  - English translation with Whisper `task="translate"`
- `target_language` is any other mapped language
  - original transcript with Whisper `task="transcribe"`
  - translated transcript with NLLB

This keeps the English path simple and uses a dedicated multilingual translation model for other target languages.

## Supported Non-English Translation Codes

The current NLLB mapping in the repo supports:

- `ar`
- `de`
- `en`
- `es`
- `fr`
- `hi`
- `it`
- `ja`
- `ko`
- `pt`
- `ru`
- `zh`

If you request another target language for the NLLB stage, you will need to add its mapping in [transcribe.py](/d:/TG/whisper/transcribe.py).

## Output Files

For input `sample.wav`, possible outputs are:

- `sample.txt`: original transcript
- `sample.en.txt`: English translation
- `sample.hi.txt`: Hindi translation
- `sample.fr.txt`: French translation

Each line keeps timestamps:

```text
[0.56s -> 2.04s] Hello everyone
[2.04s -> 5.40s] Welcome to the video
```

For NLLB translation, the timestamp prefix is preserved and only the text part is translated.

## Runtime Status

The script prints:

- requested source language
- requested target language
- chosen device mode
- model loading stage
- whether GPU fallback happened
- detected source language
- progress lines as segments are written
- periodic “still working” status during long waits
- total processing time

## Device Behavior

Device argument:

- `auto`: try NVIDIA CUDA first, otherwise use CPU
- `cpu`: force CPU
- `cuda`: force NVIDIA GPU

Important note:

- GPU acceleration in this repo is for NVIDIA CUDA environments
- on systems without working CUDA, `auto` falls back to CPU

## What `-X utf8` Does

Use:

```powershell
py -X utf8 transcribe.py sample.wav
```

This forces UTF-8 mode in Python, which helps avoid garbled non-English output on Windows terminals.

## Example Workflows

### Chinese audio to Chinese transcript

```powershell
.\extract_audio.bat .\sample.mp4
py -X utf8 transcribe.py sample.wav zh medium auto
```

### Chinese audio to English transcript

```powershell
.\extract_audio.bat .\sample.mp4
py -X utf8 transcribe.py sample.wav zh medium auto en
```

### Chinese audio to Hindi transcript

```powershell
.\extract_audio.bat .\sample.mp4
py -X utf8 transcribe.py sample.wav zh medium auto hi
```

### Auto-detect source language, output French

```powershell
.\extract_audio.bat .\sample.mp4
py -X utf8 transcribe.py sample.wav auto medium auto fr
```

## Common Notes

- English translation runs an additional Whisper pass, so it takes longer than transcript-only mode.
- Non-English translation loads a second model, so the first translation run may download more files.
- CPU-only translation can be noticeably slow, especially for larger Whisper models and long videos.
- If the detected source language matches the requested target language, the repo keeps the original transcript as the final output.

## Current Limitations

- NLLB language coverage in this repo is limited to the codes mapped in `transcribe.py`
- no `.srt` output yet
- no batch folder processing yet
- no speaker diarization
- no GUI

## Good Next Improvements

- add more NLLB language mappings
- add `.srt` export
- add a one-command end-to-end runner
- add batch processing for multiple videos
- add translation progress batching for faster non-English translation
