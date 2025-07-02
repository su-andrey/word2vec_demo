import requests
import blosc
import numpy as np
import time
from pathlib import Path
file_path = Path(__file__).parent / "test_2.txt"
all_words = []
with open(file_path, "r", encoding='utf-8') as file:
    for line in file:
        all_words.extend(line.split()) # Перегоняю тестовый файл в список слов, значительно удобнее, чем хард
req = {"target": all_words}
start_time = time.time()  # отсекаем начало выполнения операции
response = requests.post("http://localhost:8000/vectors", json={"target": ["man"]})
if response.headers["compression"] == "zstd":  # Если применяли сжатие, то нужно применить алгоритм
    data = np.frombuffer(blosc.decompress(response.content), dtype=np.float32)
else:  # Если не сжималось, то просто забираем содержимое ответа
    data = np.frombuffer(response.content, dtype=np.float32)
print(data)
end_time = time.time()  # Отсекаем конец выполнения, тем самым получая время работы
print(f"Время выполнения: {end_time - start_time} секунд")
start_time = time.time()
response = requests.post("http://localhost:8000/vectors", json={"target": all_words})
if response.headers["compression"] == "zstd":
    data = np.frombuffer(blosc.decompress(response.content), dtype=np.float32)
else:
    data = np.frombuffer(response.content, dtype=np.float32)
print(data)
end_time = time.time()
print(f"Время выполнения: {end_time - start_time} секунд")