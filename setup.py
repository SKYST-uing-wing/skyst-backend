import os
import pickle

from numpy import ndarray
from numpy import dot, ndarray
from numpy.linalg import norm
from helper import get_vec
from wav2vec import load_celebrity


folder_path = './vectors'
filenames = os.listdir(folder_path)
filenames_no_ext = [os.path.splitext(
    f)[0] for f in filenames if os.path.isfile(os.path.join(folder_path, f))]


celeb = load_celebrity()
sims = {celeb_name: [] for celeb_name, _ in celeb.items()}
for name in filenames_no_ext:
    vec1 = get_vec(name)
    for celeb_name, vec2 in celeb.items():
        sims[celeb_name].append(
            dot(vec1.flatten(), vec2.flatten())/(norm(vec1)*norm(vec2)))

for celeb_name, li in sims.items():
    print(celeb_name, sum(li)/len(li))

with open('./celebrity_bias.pickle', 'wb') as fr:
    pickle.dump({celeb_name: sum(li)/len(li)
                for celeb_name, li in sims.items()}, fr)
