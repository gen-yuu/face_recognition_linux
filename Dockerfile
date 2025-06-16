# ベースイメージをRaspberry Pi OS互換のARMv7アーキテクチャに指定
# Python 3.9 がプリインストールされている公式イメージを使用
FROM --platform=linux/arm/v7 arm32v7/debian:bullseye-slim

ENV TZ=Asia/Tokyo \
    PIP_INDEX_URL=https://www.piwheels.org/simple

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        pkg-config \
        python3            \ 
        python3-pip        \ 
        python3-setuptools \
        python3-wheel      \
        python3-dev \
        python3-venv \
        python3-opencv \
        python3-pil \
        libopenblas0-pthread \
        libgfortran5 \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコピー
COPY ./src ./src
