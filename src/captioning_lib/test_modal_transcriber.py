# To test the Modal ASR service, run the following command:
# python test_modal_transcriber.py

# See deploy_modal_transcriber.py for instructions on how to deploy the ASR service on Modal and
# how to configure Modal.

from captioning_lib import deploy_modal_transcriber
import numpy as np
import modal

asr_cls = modal.Cls.from_name(deploy_modal_transcriber.MODAL_APP_NAME, "FasterWhisper")
# asr_cls = modal.Cls.from_name(deploy_modal_transcriber.MODAL_APP_NAME, "NemoASR")
print("ASRModel class loaded from modal:", asr_cls)

# just transcribe
audio_chunk = np.random.rand(16000).astype(np.float32)
s = asr_cls().transcribe.remote(audio_chunk, None)
print("Transcription result:", s)

# translate
random_audio_np = np.random.rand(16000).astype(np.float32)
s = asr_cls().transcribe.remote(random_audio_np, translate_from_source_language='es')
print("Translation result:", s)