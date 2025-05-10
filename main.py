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
    np.save('./vectors/{name}.npy', vec)


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
    print(vec)
    return vec.tolist()


@app.get('/compare')
async def compare(name: str, target: str):
    vec1 = get_vec(name).flatten()
    vec2 = get_vec(target).flatten()
    print(vec1[:5], vec2[:5])
    cos_sim = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    print(cos_sim)
    return cos_sim.item()
