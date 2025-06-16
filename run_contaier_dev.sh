#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=face-recognition-dev
COMPOSE_FILE="./docker-compose.yml" 
FORCE_FLAG=false # -f / --force が渡されたかどうか
RECREATE_FLAG=false # -r / --recreate が渡されたかどうか
while [[ "${1:-}" =~ ^- ]]; do
  case "$1" in
    -f|--force)
      FORCE_FLAG=true
      shift ;;
    -r|--recreate)
      RECREATE_FLAG=true
      shift ;;
    -h|--help)
      echo "Usage: $(basename "$0") [-f|--force]"
      echo "  -f, --force   既存コンテナを削除し、必ず新規 build & up する"
      exit 0 ;;
    *)
      echo "[ERROR] 不正なオプション: $1"
      exit 1 ;;
  esac
done

if $FORCE_FLAG; then
  echo "[INFO] 既存コンテナを停止・削除し再構築します。"
  docker compose -f "$COMPOSE_FILE" down --remove-orphans || true
  docker compose -f "$COMPOSE_FILE" build --no-cache
  docker compose -f "$COMPOSE_FILE" up -d

elif $RECREATE_FLAG; then
  echo "[INFO] 既存コンテナを再起動します。"
  docker compose -f "$COMPOSE_FILE" up -d
else
  # コンテナの存在チェック（停止状態も含む）
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$SERVICE_NAME"; then
    echo "[INFO] コンテナ $SERVICE_NAME が存在しません。ビルド＆起動します。"
    docker compose -f "$COMPOSE_FILE" build 
    docker compose -f "$COMPOSE_FILE" up -d
  else
    echo "[INFO] コンテナ $SERVICE_NAME は既に存在します。"
    # 起動していなければ起動
    if ! docker ps --format '{{.Names}}' | grep -qx "$SERVICE_NAME"; then
      echo "[INFO] コンテナが停止しているため再起動します。"
      docker start "$SERVICE_NAME"
    fi
  fi
fi

# bash シェルに入る
echo "[INFO] ${SERVICE_NAME} に /bin/bash で接続します。"
docker exec -it "${SERVICE_NAME}" /bin/bash