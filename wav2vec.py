import numpy as np
import openl3
import soundfile as sf

def audio_preprocessing(audio, defaultlen = 16000 * 5):
  audio_length = audio.shape[0]
  if audio_length > defaultlen:
    return audio[:defaultlen]

  else:
    N = defaultlen // audio_length
    temp = audio
    for _ in range(N):
      temp = np.append(temp, audio)
    return temp[:defaultlen]

def wav2vec(path):
    audio, sr = sf.read(path)
    audio_preprocessed = audio_preprocessing(audio, sr*5)
    emb, ts = openl3.get_audio_embedding(audio_preprocessed, sr=sr, embedding_size=512, hop_size=0.5)
    vec = emb.flatten()
    return vec
    