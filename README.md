# Video Transcript Generator

This project extracts audio from a video file and generates a timestamped transcript using `faster-whisper`.

It now works for any language supported by Whisper, not just Chinese.

You can:

- let Whisper auto-detect the language
- force a specific language such as `zh`, `en`, `hi`, `ja`, `fr`, or `de`
- watch a live progress bar while transcription runs
- see transcript lines written into the output file as they are generated

## Features

- extracts mono `16 kHz` WAV audio from video using `ffmpeg`
- transcribes audio with `faster-whisper`
- supports any Whisper-supported language
- defaults to automatic language detection
- writes timestamped transcript output to `.txt`
- updates the transcript file live during processing
- shows clear model-loading and transcription status messages
- shows an approximate progress bar based on WAV duration

## Project Files

- [extract_audio.bat](/d:/TG/whisper/extract_audio.bat): converts input video to a same-name `.wav` file
- [transcribe.py](/d:/TG/whisper/transcribe.py): runs transcription and writes transcript output
- [requirements.txt](/d:/TG/whisper/requirements.txt): Python dependency list
- `sample.mp4`: sample input file

## Requirements

You need:

1. Python 3.11 or later
2. `ffmpeg`
3. `pip`

## Setup

### 1. Install Python

Check whether Python is available:

```powershell
python --version
```

or:

```powershell
py --version
```

If `py` does not find Python, you can still run with the full executable path, for example:

```powershell
C:\Users\ZML-WIN-VijayN-01\AppData\Local\Programs\Python\Python311\python.exe --version
```

### 2. Install ffmpeg

Check:

```powershell
ffmpeg -version
```

If this fails, install `ffmpeg` and add it to `PATH`.

### 3. Install Python dependencies

From the project folder:

```powershell
pip install -r requirements.txt
```

If needed:

```powershell
python -m pip install -r requirements.txt
```

or:

```powershell
py -m pip install -r requirements.txt
```

## Usage

### Step 1. Extract audio from a video

```powershell
extract_audio.bat sample.mp4
```

This creates:

```text
sample.wav
```

The WAV is generated as:

- mono
- `16 kHz`
- `pcm_s16le`

### Step 2. Run transcription

Basic command with auto-detection:

```powershell
py -X utf8 transcribe.py sample.wav
```

Force a language explicitly:

```powershell
py -X utf8 transcribe.py sample.wav zh
py -X utf8 transcribe.py sample.wav en
py -X utf8 transcribe.py sample.wav hi
py -X utf8 transcribe.py sample.wav ja
```

Pick a model explicitly:

```powershell
py -X utf8 transcribe.py sample.wav auto medium
py -X utf8 transcribe.py sample.wav zh large-v3
py -X utf8 transcribe.py sample.wav en base
```

If `py` is unreliable on your machine, use the direct Python executable:

```powershell
C:\Users\ZML-WIN-VijayN-01\AppData\Local\Programs\Python\Python311\python.exe -X utf8 transcribe.py sample.wav auto medium
```

## Script Arguments

The script accepts:

```text
python transcribe.py <audio_file> [language|auto] [model_size]
```

Arguments:

- `audio_file`: path to the WAV file
- `language|auto`: optional, defaults to `auto`
- `model_size`: optional, defaults to `medium`

Examples:

```powershell
py -X utf8 transcribe.py sample.wav
py -X utf8 transcribe.py sample.wav auto
py -X utf8 transcribe.py sample.wav zh
py -X utf8 transcribe.py sample.wav en small
py -X utf8 transcribe.py sample.wav hi medium
py -X utf8 transcribe.py sample.wav ja large-v3
```

Common language codes:

- `auto`: automatic language detection
- `zh`: Chinese
- `en`: English
- `hi`: Hindi
- `ja`: Japanese
- `ko`: Korean
- `fr`: French
- `de`: German
- `es`: Spanish
- `it`: Italian
- `pt`: Portuguese
- `ru`: Russian
- `ar`: Arabic

Whisper supports many more languages than the list above.

## Output

For input like:

```text
sample.wav
```

the script writes:

```text
sample.txt
```

Each output line looks like:

```text
[0.56s -> 2.04s] Hello everyone
[2.04s -> 5.40s] Welcome to the video
```

The output file is updated live while transcription is running.

## Progress and Status Messages

The script now shows:

- requested language or auto-detect mode
- model loading step
- notice when first-run model download may happen
- transcription start
- detected language
- live progress bar
- final completion message

Example:

```text
Preparing transcription for: sample.wav
Requested language: auto-detect
Selected model: medium
Step 1/3: Loading Whisper model...
If this is the first run for this model, it may download files before transcription begins.
Step 1/3 complete: Model is ready.
Step 2/3: Starting transcription...
Transcribing: sample.wav
Audio length: 150.22s
Writing transcript live to: sample.txt
Step 2/3 active: Receiving segments from the model...
Detected language: en (probability: 0.99)
[##########--------------------] 35.12% (52.77s / 150.22s)
```

## What `-X utf8` Does

`-X utf8` tells Python to force UTF-8 mode.

This is especially helpful on Windows so non-English text such as Chinese, Hindi, Japanese, Arabic, or Russian does not get mangled in console output.

Recommended:

```powershell
py -X utf8 transcribe.py sample.wav
```

## Model Notes

Supported examples in this repo:

- `tiny`
- `base`
- `small`
- `medium`
- `large-v3`

General guidance:

- `base` or `small`: faster for testing
- `medium`: balanced default
- `large-v3`: slower, but often better for harder audio

On first run, a model may need to download before transcription starts.

## Common Issues

### `ffmpeg` is not recognized

Install `ffmpeg` and add it to `PATH`.

### `py` says `No installed Python found`

Use the full Python path or repair the Python installation.

Example:

```powershell
C:\Users\ZML-WIN-VijayN-01\AppData\Local\Programs\Python\Python311\python.exe -X utf8 transcribe.py sample.wav auto medium
```

### Transcript file stays empty at first

Possible reasons:

- model is still downloading
- model is still loading
- transcription has not begun producing segments yet

Watch the terminal status messages to see which stage is currently active.

### Non-English characters look broken

Use:

```powershell
py -X utf8 transcribe.py sample.wav
```

and make sure your editor or terminal is using UTF-8 properly.

## Example Workflows

### Chinese transcription

```powershell
extract_audio.bat sample.mp4
py -X utf8 transcribe.py sample.wav zh medium
```

### English transcription

```powershell
extract_audio.bat sample.mp4
py -X utf8 transcribe.py sample.wav en medium
```

### Hindi transcription

```powershell
extract_audio.bat sample.mp4
py -X utf8 transcribe.py sample.wav hi medium
```

### Automatic language detection

```powershell
extract_audio.bat sample.mp4
py -X utf8 transcribe.py sample.wav auto medium
```

## Current Limitations

This repo is still intentionally simple. It does not yet include:

- subtitle export like `.srt`
- batch processing for many files
- speaker diarization
- translation output
- punctuation cleanup or post-processing
- GUI

## Good Next Improvements

- add `.srt` export
- derive output names automatically from input files
- add better exception handling for model download/runtime failures
- add one-click end-to-end batch commands
- optionally support translation into English
