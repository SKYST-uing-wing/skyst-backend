from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from numpy import dot, ndarray
from numpy.linalg import norm
import shutil
import os

from helper import calc_cos_sim, convert_and_save_vec, get_vec, load_celebrity_bias
from wav2vec import get_spectrogram, load_celebrity, wav2vec

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"] 등 필요한 도메인만 지정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-mp3")
async def upload_mp3(name: str, file: UploadFile = File(...)):
    print(name)
    if not file.filename.endswith(".wav"):
        return JSONResponse(
            status_code=400,
            content={"message": "Only MP3 files are allowed."}
        )

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    convert_and_save_vec(name)

    return {"message": "File uploaded successfully", "filename": file.filename}


@app.get('/result')
async def result(name: str):
    vec1 = get_vec(name)
    celeb: dict[str, ndarray] = load_celebrity()
    celeb_bias: dict[str, float] = load_celebrity_bias()
    sims = {celeb_name: dot(vec1.flatten(), vec2.flatten(
    ))/(norm(vec1)*norm(vec2))-celeb_bias[celeb_name] for celeb_name, vec2 in celeb.items()}
    sim_celeb = max(celeb, key=lambda x: sims[x])
    return {'vectors': vec1.tolist(), 'similar_celeb': {'name': sim_celeb, 'cos_sim': sims[sim_celeb]}}


@app.get('/compare')
async def compare(name: str, target: str):
    cos_sim = calc_cos_sim(name, target)
    return {'cos_sim': cos_sim.item()}


@app.get('/compareall')
async def compareall(name: str):
    folder_path = './vectors'
    filenames = os.listdir(folder_path)
    filenames_no_ext = [os.path.splitext(
        f)[0] for f in filenames if os.path.isfile(os.path.join(folder_path, f))]
    ret = {target: calc_cos_sim(name, target)
           for target in filenames_no_ext if target != name}
    return {'list_cos_sim': ret}


@app.get('/spectrogram')
async def spectrogram(name: str):
    return get_spectrogram(f'./uploads/{name}.wav')
