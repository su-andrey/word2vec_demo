FROM python:3.10-slim
# Используем только python, причём облегчённый

RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*
# Разбивать на 3 RUN интернет не рекомендует, так как в таком случае будет создаваться 3 отдельных слоя, вместо одного
# Что повлияет на время сборки образа.
# Этой инструкцией обновляю список версий, устанавливаю gcc и библиотеки python (необходимо для СБОРКИ gensim)
# Удаляю кеш файлы от установки, необязательно, но требование "запустить в полях" подразумевает экономию места
WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY model/GoogleNews-vectors-negative300.bin.gz ./model/
# Копирую только необходимые файлы, тесты, например, в докер копировать неосмысленно

RUN pip install --no-cache-dir -r requirements.txt && apt-get remove -y gcc python3-dev && apt-get autoremove -y
# Устанавливаем зависимости по requirements (без кэша), затем удаляем более не нужные gcc и файлы библиотек,
# Они требовались только для первичной сборки библиотек, поэтому после их можно удалять для экономии памяти
# Автоматиечески удаляем неиспользуемые пакеты - уменьшение размера образа

ENV PATH=/app/model/GoogleNews-vectors-negative300.bin.gz
EXPOSE 8000
CMD ["/usr/local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Без полного пути uvicorn не работает, 400 образ при запуске