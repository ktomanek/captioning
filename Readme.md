# On-Device Captioning

Supported ASR models:

* [FasterWhisper](https://github.com/SYSTRAN/faster-whisper)
* [Whisper ONNX](https://huggingface.co/docs/transformers/serialization#onnx) - Custom Whisper models exported to ONNX format (requires 3 files: encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx)
* [Moonshine ONNX](https://github.com/usefulsensors/moonshine)
* [NVidia Nemo FastConformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)


Other models can easily added by adding a ```Transcriber``` class.

## Running Non-Local models

While this package aims for running small models that can run locally, it also includes a Transcriber implementation using a mix of local and remote processing (partial segments are transcribed with a fast, local model and once a segment is finished a larger model hosted on GPU will process this segment for better quality.)

We're using [Modal](https://modal.com/docs/guide) here to enable on-demand usage of GPUs with pre-deployed functions. See `deploy_modal_transcriber.py`. Use `--model=remote` or `--model=remote_and_local`.

# Installation

Install library

```pip install -e .```

Base installation (will work with faster-whisper models)

```pip install -r requirements.txt``

You may have to install:

```sudo apt-get install portaudio19-dev```


Download the silero VAD model

```python src/captioning_lib/helpers/download_silero_vad_model.py```


## Optional: other models 

To install Moonshine ONNX models (recommended):

```pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx```

To install Nemo models (optional)

```pip install "nemo_toolkit[asr]"```

To install the Vosk model (optional)

```pip install vosk```

## Optional: Run models remotely

To use the Modal based transcriber, (optional):

```pip install modal==1.0```

and then run

```modal setup```

## Optional: Whisper HF to ONNX conversion

* if you want to use onnx Whisper models (for `-m whisperonnx` model type), you need to convert your Whisper model to onnx first;
* you can use `src/captioning_lib/convert_hf_whisper_to_onnx.py` for that, this will create a f32 and a int8 version.
* you will need to `pip install optimum` for that.

# Running Captioning App

You can run the captioning app directly via:

```python src/captioning_lib/captioning_app.py --model fasterwhisper_tiny --rich_captions --eos_min_silence=200```

If your input device doesn't support recording in 16khz, you can use the ALSA device string or ALSA PCM device name instead of the index of your input device (Format: plughw:CARD,DEVICE).
In this case, the plughw:1,0 device will automatically resample from your device's native sampling rate to the requested sampling rate (16kHz).

```
python src/captioning_lib/captioning_app.py -m fasterwhisper_tiny -rc -i plughw:1,0
```

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
  * non-streaming: ```python transcription_app.py --audio_file=../samples/jfk_asknot.wav --reference_file=../samples/jfk_asknot.txt  --model XXXX```  


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
* mem - memory footprint

Hardware tested

* Rasp - Raspberry Pi 5, 16GB
* MiniPC - Beelink Mini S, N100, 16GB
* Mac M2 - Macbook Air M2, 16GB

|                         | mem | WER         || RTF Mac M2  || RTF MiniPC  || RTF Rasp ||
| --                      | -- |  -- | --    |  --   | --  |    -- | --  |    -- | --|
| model                   | -   | stream   | offl  |  stream| offl  |   stream| offl    |     stream|   offl |
| moonshine-onnx-tiny     | ~550 MB | 0.23 | 0.0 | 65.2 | 50.6 | 20.9 | 11.6 | 15.1 | 23.5 | 
| moonshine-onnx-base     | ~970 MB | 0.16 | 0.0 | 32.9 | 16.1 | 12.6 | 5.1 | 7.8  | 12.7 | 
| fasterwhisper-tiny      | ~230 MB | 0.25 | 0.0 | 6.0  | 34.7 | 2.6  | 22.1 | 1.1  | 6.6|
| fasterwhisper-base      | ~320 MB | 0.11 | 0.0 | 4.0  | 17.2 | 1.8  | 14.1 | 0.8  | 4.3 |
| fasterwhisper-small     | ~640 MB | 0.10 | 0.0 | 1.3  | 7.3  | 0.7  | 4.8  | 0.1  | 1.3 |
| whisper-onnx-tiny (int8)| ~370 MB | - | - | 0.0  | -  | -  | -  | 3.3 | 15.78
| nemo-fastconformer-ctc  | ~1440 MB | 0.14 | 0 | 17.1 | 55.8 | 8.2  | 19.2 | 4.3  | 11.1 |
| nemo-fastconformer-rnnt | ~1440 MB | 0.13 | 0.05 | 15.2 | 53.4 | 6.9  | 12.7 | 3.1  | 8.14 | 
| vosk-tiny               | ~110 MB | 0.31 | 0.18 | 19.2 | 25.8 | | | | |



## Take-aways

* Raspberry Pi 5 can run all tested models (except for fasterwhisper-small and base)  on device, not all are practically acceptable for real-time streaming for faster speakers. Moonshine-onnx-tiny and the onnx version of whisper make the cut and seems ok for real time scenarios.
* In general, Moonshine models significantly faster than tested Nemo models with much lower memory footprint (due to ONNX opt and smaller parameter size), but have higher WER (see HF leaderboard)


# Good Settings depending on hardware

## Different Modes for Partial Transcriptions

The system supports two modes for partial transcriptions:

### Default Mode (Retranscribes all audio chunks in buffer)
Retranscribes all accumulated audio for each partial. Provides better quality and context for transcription, especially with short partial durations.

```bash
python captioning_app.py --model whisperonnx --model_path /path/to/models/ --min_partial_duration 0.5
```
- **Strategy**: Frequent updates with full retranscription of partial (default)
- **Benefits**: Best quality, very responsive
- **Trade-off**: Higher CPU/GPU usage (acceptable on fast hardware), increasing computational cost as partials grow

--> recommended on faster hardware

### Recent-chunk Mode (`--recent_chunk_mode`, only transcribes audio chunk since last partial transcription)  
Transcribes only the most recent audio chunk for partials, then retranscribes the entire segment for final results. More efficient for longer partial durations but requires sufficient context per chunk. Enables token-by-token streaming output for supported transcribers (whisperonnx, to some extend fasterwhisper).

```bash
python captioning_app.py --model whisperonnx --model_path /path/to/models/ --min_partial_duration 2.0 --recent_chunk_mode
```
- **Strategy**: Recent-chunk mode with longer duration
- **Benefits**: Reduces expensive encoding overhead while maintaining token streaming
- **Performance**: Consistent computational cost per chunk

--> good strategy on slowe hardware, like a weak CPU, on a Raspberry Pi etc

### General Guidelines

- **Default behavior**: Retranscribe mode with short partials (good quality, more computation)
- **For efficiency**: Use `--recent_chunk_mode` with longer durations (> 2s)
- **Token streaming**: Real-time word-by-word output is enabled with `--recent_chunk_mode` for models supporting autoregressive decoding (fasterwhisper, whisperonnx)
- **Resource constraints**: Use `--recent_chunk_mode` with higher `--min_partial_duration` to reduce computational load

# Translations

the ```TranscriptionTranscriber``` can transcribe from a given source language to English. It uses a tiny Whisper model for partial transcriptions and a large Whisper model running on Modal for the final translations. 

When running with the ```captioning_server.py``` there is a specific translation client that will show the original transcript (source language) and the translations (English) in separate windows.

Since Whisper (esp tiny) doesn't work very well on non-EN, it is recommended to increase the transcription buffer (by setting --min_partial_duration):

```python captioning_server.py -m translation_from_de  --min_partial_duration 0.5```

# TODO

- [ ] Add conversion script for exporting Whisper models to ONNX format with the required 3-file structure (encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx)