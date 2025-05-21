# On-Device Captioning

Supported ASR models:

* [FasterWhisper](https://github.com/SYSTRAN/faster-whisper)
* [Moonshine ONNX](https://github.com/usefulsensors/moonshine)
* [NVidia Nemo FastConformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)


Other models can easily added by adding a ```Transcriber``` class.

## Running Non-Local models

While this package aims for running small models that can run locally, it also includes a Transcriber implementation using a mix of local and remote processing (partial segments are transcribed with a fast, local model and once a segment is finished a larger model hosted on GPU will process this segment for better quality.)

We're using [Modal](https://modal.com/docs/guide) here to enable on-demand usage of GPUs with pre-deployed functions. See `deploy_modal_transcriber.py`. Use `--model=remote` or `--model=remote_and_local`.

# Installation

Base installation (will work with faster-whisper models)

```pip install -r requirements.txt``

You may have to install:

```sudo apt-get install portaudio19-dev```

To install Nemo models

```pip install "nemo_toolkit[asr]"```

To install Moonshine ONNX models:

```pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx```

To install the Vosk model:

```pip install vosk```

To use the Modal based transcriber, also install:

```pip install modal```

and then run

```modal setup```

# Running Captioning App

You can run the captioning app directly via:

```python captioning_app.py --model moonshine_onnx_tiny --rich_captions --eos_min_silence=200```

# Evaluate ASR Streamig performance

The `captioning_app.py` allows to feed in an audio file along with the corresponding transcriptions. It will then simulate streaming on this file and calculate the WER based on full segments (partials are not evaluated).

Example:

```python captioning_app.py --model moonshine_onnx_tiny --eval --audio_file=../samples/jfk_space.wav --reference_file=../samples/jfk_space.txt```

# Running Captioning Server

Captioning can also be run in a client-server mode. The server runs on Flask, the client connects via Websockets.

See ```captioning_server.py``` and ```captioning_client.py```

Currently the server does not handle concurrent connections.

# Streaming Performance Comparison

## Setting

* WER and RTFx are only calculated for full segments, not partials (those are skipped here)
* WER is calculated after normalization (no punctuation and casing)
* for all experiments, the following settings were used:
  * streaming: ```python captioning_app.py --eval --audio_file=../samples/jfk_space.wav --reference_file=../samples/jfk_space.txt --eos_min_silence=100 --rtf=0.0 --max_segment_duration=10 --model XXXX```
  * non-streaming: ```python transcription_app.py --audio_file=../samples/jfk_space.wav --reference_file=../samples/jfk_space.txt  --model XXXX```  


## Notes


* this is not meant as general performance evaluation of the respective ASR models -- see, e.g., https://huggingface.co/spaces/hf-audio/open_asr_leaderboard for that instead
    * but instead it compares WER and RTFx on different hardware in streaming mode
    * nemo models transcribe numbers as words, leading to higher WER too
    * overall, especially in the non-streaming scenario most errors are due to normalization of numbers
* moonshine-onnx-base had strange issues in the non-streaming scenario, leading to lots of dropped parts of the audio file, unclear why

Overall:

Since WER is mostly driven by streaming scenario and also unit normalization (and also this is a test on a single audio file only), the WER results by themselves aren't very meaningful and don't allow to compare models between each other too well. 

-> for WER, a comparison between streaming and non-streaming modes makes sense and tells about the drop we can expect when going to a streaming scenario (obvisouly different streaming settings will impact WER differently)

-> RTFx can be compared across models and modes meaningfully


## Results

modes:

* stream - streaming
* offl - offline / non-streaming

Hardware tested

* Rasp - Raspberry Pi 5, 16GB
* MiniPC - Beelink Mini S, N100, 16GB
* Mac M2 - Macbook Air M2, 16GB

|                         | WER         || RTF Mac M2  || RTF MiniPC  || RTF Rasp ||
| --                      |  -- | --    |  --   | --  |    -- | --  |    -- | --|
| model                   | stream   | offl  |  stream| offl  |   stream| offl    |     stream|   offl |
| moonshine-onnx-tiny     | 0.23 | 0.38 | 65.2 | 50.6 | 20.9 | 11.6 | 15.1 | 6.3 | 
| moonshine-onnx-base     | 0.16 | -- | 32.9 | 16.1 | 12.6 | 5.1 | 7.8  | 1.9 | 
| whisper-tiny            | 0.25 | 0.09 | 6.0  | 34.7 | 2.6  | 22.1 | 1.1  | 7.1 |
| whisper-base            | 0.11 | 0.08 | 4.0  | 17.2 | 1.8  | 14.1 | 0.8  | 3.5 |
| whisper-small           | 0.10 | 0.06 | 1.3  | 7.3  | 0.7  | 4.8  | 0.1  | 1.0 |
| nemo-fastconformer-ctc  | 0.14 | 0.14 | 17.1 | 55.8 | 8.2  | 19.2 | 4.3  | 6.6 |
| nemo-fastconformer-rnnt | 0.13 | 0.15 | 15.2 | 53.4 | 6.9  | 12.7 | 3.1  | 4.6 | 
| vosk-tiny               | 0.31 | 0.18 | 19.2 | 25.8 | | | | |


## Take-aways

* Raspberry Pi 5 can run all tested models (except for whisper-small and base)  on device in acceptable speed for streaming for scenarios where the speaker is on the slower side (see `../sample/jfk_space.wav`); for faster speech, only the moonshine_onnx_tiny model seems to be fast enough.
* In general, Moonshine models significantly faster than tested Nemo models with much lower memory footprint (due to ONNX opt and smaller parameter size), but have higher WER (see HF leaderboard)


# Translations

the ```TranscriptionTranscriber``` can transcribe from a given source language to English. It uses a tiny Whisper model for partial transcriptions and a large Whisper model running on Modal for the final translations. 

When running with the ```captioning_server.py``` there is a specific translation client that will show the original transcript (source language) and the translations (English) in separate windows.

Since Whisper (esp tiny) doesn't work very well on non-EN, it is recommended to increase the transcription buffer (by setting --min_partial_duration):

```python captioning_server.py -m translation_from_de  --min_partial_duration 0.5```