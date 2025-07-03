import blosc
import numpy as np
from fastapi.testclient import TestClient

from main import app, find_vector

client = TestClient(app)


def test_one_word():
    response = client.post("/vectors", json={"target": ["man"]})
    assert response.status_code == 200  # Ответ успешно получен
    assert response.headers["compression"] == "none"  # Сжатия нет
    data = np.frombuffer(response.content, dtype=np.float32)
    assert data.shape == (300,)  # Размерности подходящие


def test_incorrect_data():
    response = client.post("/vectors", json={"somestrangekey": ["man"]})
    assert response.status_code == 422  # Валидация не должна допускать такие данные до выполнения функции
    response = client.post("/vectors", json={"target": "some weird list that looks like a string"})
    assert response.status_code == 422  # Валидация не должна допускать такие данные до выполнения функции


def test_words_uncompressed():
    response = client.post("/vectors", json={
        "target": ["man", "king", "queen", "woman", "ghbdklasdasssxztn", "chair", "table", "PHP"]})
    assert response.status_code == 200  # Ответ получен успешно
    assert response.headers["compression"] == "none"  # Для сжатия недостаточно слов
    data = np.frombuffer(response.content, dtype=np.float32)
    assert data.shape == (7 * 300,)  # Одно слово несуществующее, должно отпасть


def test_fake_words():
    response = client.post("/vectors", json={"target": ["JAalsdfkasa", "фываолдж", "蒸謬船哉汽"]})
    assert response.status_code == 200  # Слова выдуманные, но ошибки быть не должно
    assert response.content == b''  # Возвращено должно быть пустое значение
    assert response.headers["compression"] == "none"  # Логично, что не сжатое


def test_many_fake_words():
    response = client.post("/vectors", json={"target": ["JAalsdfkasa", "фываолдж", "蒸謬船哉汽"] * 10000})
    assert response.status_code == 200
    assert response.content == b''
    assert response.headers["compression"] == "none"  # Слова все выдуманные, поэтому без сжатия


def test_compression_many_words():
    word_list = ["man", "king", "woman", "queen"] * 5001  # Достаточно слов для сжатия
    response = client.post("/vectors", json={"target": word_list})
    assert response.status_code == 200
    assert response.headers["compression"] == "zstd"  # Проверяем, что ответ сжат
    decompressed = blosc.decompress(response.content)
    data = np.frombuffer(decompressed, dtype=np.float32)
    assert data.shape == (4 * 5001 * 300,)  # Сверяю размер


# Отдельно протестируем функцию получения вектора на некоторые базовые ошибки

def test_vector_cases():
    assert find_vector("аничжум") is None
    assert find_vector("蒸謬船哉汽") is None
    assert find_vector("sadnaptqyp") is None  # всякие несуществующие слова не должны найтись
    assert find_vector("man") is not None
    assert find_vector("I") is not None  # а существующие должны


def test_vector_shape():
    res = find_vector("man")
    assert res is not None
    assert res.shape == (300,)
    assert res.dtype == np.float32  # Результат есть, содержит 300 измерений нужной размерности
