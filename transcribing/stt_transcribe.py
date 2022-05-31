import os
import ctc_decoders
import torchaudio
import soundfile as sf

import numpy as np
# Import audio processing library
import librosa

def get_stt_transcription(model, file):
    # get stt transcription using the pretrained model
    transcript = model.transcribe(paths2audio_files=file)[0]
    return transcript

