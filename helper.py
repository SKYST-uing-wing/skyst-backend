
from functools import lru_cache
from pathlib import Path
import pickle

from fastapi import HTTPException
from numpy import dot, ndarray
from numpy.linalg import norm
import numpy as np

from wav2vec import wav2vec


def convert_and_save_vec(name):
    file_path = Path(f'./uploads/{name}.wav')
    vec = wav2vec(file_path)
    np.save(f'./vectors/{name}.npy', vec)


def get_vec(name):
    file_path = Path(f'./vectors/{name}.npy')
    if file_path.exists():
        return np.load(file_path)
    file_path = Path(f'./uploads/{name}.wav')
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    convert_and_save_vec(name)
    return get_vec(name)


def calc_cos_sim(name, target):
    vec1 = get_vec(name).flatten()
    vec2 = get_vec(target).flatten()
    cos_sim = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    return cos_sim


@lru_cache
def load_celebrity_bias():
    with open('./celebrity_bias.pickle', 'rb') as fr:
        celebrity_bias = pickle.load(fr)
    return celebrity_bias
