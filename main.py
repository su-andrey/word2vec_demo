from functools import lru_cache
from typing import List
import os
import blosc
import numpy as np
from fastapi import FastAPI, Response
from gensim.models import KeyedVectors
from pydantic import BaseModel

app = FastAPI()
PATH = os.getenv("PATH")
model = KeyedVectors.load_word2vec_format(PATH, binary=True)
MIN_COMPR_VAL = 10000  # Количество слов, которые можно обработать, начиная с которого применяем сжатие


class Batch(BaseModel):  # Простейший класс, описывающий (валидирующий) входные данные
    target: List[str]


@lru_cache(
    maxsize=25000)  # Взял не очень большое значение, чтобы не тратить избыточно памяти, думаю для самых популярных слов хватит
def find_vector(word):
    try:
        return model[word]  # Преобразуем все слова в вектора, при этом отлавливая исключения (отсутствие слов)
    except KeyError:
        return None


@app.post("/vectors")
async def get_vectors_from_words(request: Batch):
    vectors = []
    for word in request.target:
        current = find_vector(word)
        if current is not None:
            vectors.append(current)
    if len(vectors) == 0:
        return Response(content=b'', media_type="application/octet-stream",
                        headers={"compression": "none"})  # Не нашли никаких слов
    vec_array = np.array(vectors, dtype=np.float32)  # преобразуем в numpy array для удобства и скорости
    if len(vectors) > MIN_COMPR_VAL:  # если векторов много, используем сжатие (понятие много задаётся константой)
        compressed = blosc.compress(vec_array.tobytes(), typesize=4, clevel=9,
                                    cname="zstd")  # Сжимаем, оставляя слово длинной 32, максимальной степенью сжатия, алгоритмом Zstandart
        return Response(content=compressed, media_type="application/octet-stream",
                        headers={"compression": "zstd"})  # В заголовке передаём информацию о сжатии
    return Response(content=vec_array.tobytes(), media_type="application/octet-stream",
                    headers={"compression": "none"})  # В заголовке передаём информацию об отсутствии сжатия
