import requests
import blosc
import numpy as np
import time
from pathlib import Path
file_path = Path(__file__).parent / "test_3.txt"
all_words = []
with open(file_path, "r", encoding='utf-8') as file:
    for line in file:
        all_words.extend(line.split())
req = {"target": all_words}
start_time = time.time()
response = requests.post("http://localhost:8000/compressed/batch", json=req)
res = np.frombuffer(blosc.decompress(response.content), dtype=np.float32)
print(len(res)) # Проверить на глаз, что количество совпадает в обоих способах
end_time = time.time()
print(f"Время выполнения: {end_time - start_time} секунд")
start_time = time.time()
response = requests.post("http://127.0.0.1:8000/batch", json=req)
data = np.frombuffer(response.content, dtype=np.float32)
print(len(data))
end_time = time.time()
print(f"Время выполнения: {end_time - start_time} секунд")
