FROM python:3.13-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev build-essential && \
    apt-get clean

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]

