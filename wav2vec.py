import numpy as np
import openl3
import soundfile as sf
from sklearn.decomposition import PCA
import pickle


def audio_preprocessing(audio, defaultlen=16000 * 5):
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
    emb, ts = openl3.get_audio_embedding(
        audio_preprocessed, sr=sr, embedding_size=512, hop_size=0.5)

    emb_tot = np.load("./emb_tot_test_512_arr.npy")

    vec = []
    for i in range(10):
        pca = PCA(n_components=5)
        pca.fit(emb_tot[:, i, :])

        amp = [audio_preprocessed[(sr//2)*i: (sr//2)*(i+1)].max()]

        pca_output = list(pca.transform(emb[i].reshape(1, -1))[0])

        vec.append(amp + pca_output)

    return np.array(vec)

def load_celebrity():
    with open('./celebrity.pickle', 'rb') as fr:
        celebrity = pickle.load(fr)

    return celebrity