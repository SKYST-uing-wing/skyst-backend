from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from numpy import dot
from numpy.linalg import norm
import numpy as np


import shutil
import os

from wav2vec import wav2vec

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


@app.get('/result')
async def result(name: str):
    vec = get_vec(name)
    return {'vectors': vec.tolist()}


@app.get('/compare')
async def compare(name: str, target: str):
    cos_sim = calc_cos_sim(name, target)
    return {'cos_sim': cos_sim.item()}


def calc_cos_sim(name, target):
    vec1 = get_vec(name).flatten()
    vec2 = get_vec(target).flatten()
    cos_sim = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    return cos_sim


@app.get('/compareall')
async def compareall(name: str):
    folder_path = './vectors'
    filenames = os.listdir(folder_path)
    filenames_no_ext = [os.path.splitext(
        f)[0] for f in filenames if os.path.isfile(os.path.join(folder_path, f))]
    ret = {target: calc_cos_sim(name, target)
           for target in filenames_no_ext if target != name}
    return {'list_cos_sim': ret}
