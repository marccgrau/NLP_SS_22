
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def get_stt_transcription(model, file):
    # get stt transcription using the pretrained model
    transcript = model.transcribe(paths2audio_files=file)[0]
    return transcript

def get_stt_probs(model, file):
    logits = model.transcribe(paths2audio_files=file, logprobs=True)[0]
    probs = softmax(logits)
    return probs
