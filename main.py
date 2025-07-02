from typing import List
import blosc
import numpy as np
from fastapi import FastAPI, Response
from gensim.models import KeyedVectors
from pydantic import BaseModel

app = FastAPI()
path = 'model/GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(path, binary=True)


class Word(BaseModel):
    target: str  # Можно, конечно, сделать через fastapi.Body(), но так сразу удобно валидируется вход


@app.post("/compressed")  # Сделал прямо на главной страничке, тк не было дополнительных требований
async def compressed_vector(word: Word):
    target_word = word.target  # Считываем слово на входе
    try:
        vec = model[target_word]  # Находим его вектор
    except KeyError:
        return Response(content=b'', media_type="application/octet-stream")
    compressed_vec = blosc.compress(vec.tobytes(), typesize=4, clevel=9,
                                    cname="zstd")  # сжимаем, для этого преобразуем в последовательность  байтов,
    # оставяем слово 4 байта (чтобы не потерять точность), используя алгоритм zstandart
    return Response(compressed_vec,
                    media_type="application/octet-stream")  # Здесь указываем тип данных, так как иначе получаем 500 ошибку


@app.post("/")
async def vector(word: Word):
    target_word = word.target
    try:
        vec = model[target_word]
    except KeyError:
        return Response(content=b'', media_type="application/octet-stream")
    res_coord = vec.tolist()  # Здесь я преобразую его в список, тк сразу вернуть не получается,
    # по-хорошему от этого стоит избавиться
    return res_coord


class Batch(BaseModel):
    target: List[str]


@app.post("/batch")
async def batch_vector(batch: Batch):
    vectors = []
    for word in batch.target:
        try:
            vectors.append(model[word])
        except KeyError:
            continue  # Пропускаем отсутствующие слова
    if not vectors:
        return Response(content=b'', media_type="application/octet-stream")
    vectors_array = np.array(vectors, dtype=np.float32)
    return Response(content=vectors_array.tobytes(),media_type="application/octet-stream")


@app.post("/compressed/batch")
async def compressed_batch_vector(batch: Batch):
    arrays = []
    for target_word in batch.target:
        try:
            arrays.append(model[target_word])
        except KeyError:
            pass
    if len(arrays) == 0:
        return Response(content=b'', media_type="application/octet-stream")
    vectors_array = np.array(arrays, dtype=np.float32)
    compressed_vec = blosc.compress(vectors_array.tobytes(), typesize=4, clevel=9, cname="zstd")
    return Response(compressed_vec, media_type="application/octet-stream")
