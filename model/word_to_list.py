from pathlib import Path
file_path = Path(__file__).parent / "test.txt"
all_words = []

with open(file_path, "r", encoding='utf-8') as file:
    for line in file:
        all_words.extend(line.split())
print(f"Успешно загружено {len(all_words)} слов")
print("Пример первых 10 слов:", all_words[:10])

