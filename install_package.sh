#!/usr/bin/env bash

# スクリプトが失敗した時点で直ちに終了する
set -e

# 引数が指定されていない場合のデフォルトのYAMLファイルパス
DEFAULT_YAML_FILE="$(dirname "$0")/system_packages.yml"

# コマンドライン引数でYAMLファイルが指定されていればそれを使用し、なければデフォルト値を使用
YAML_FILE="${1:-$DEFAULT_YAML_FILE}"

# yqコマンドのinstall
if ! command -v yq &> /dev/null; then
    echo "Warning: yq コマンドが見つかりません."
    echo "yqコマンドをインストールします..."
    sudo apt install yq -y
fi

# YAMLファイルの存在確認
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: YAMLファイルが見つかりません: $YAML_FILE"
    exit 1
fi

# --- メイン処理 ---
echo "YAMLファイルからパッケージリストを読み込みます: $YAML_FILE"
# yq を使ってYAMLから全てのパッケージ名を抽出し、改行で連結する
PACKAGES_TO_INSTALL=$(yq '.[] | .[]' "$YAML_FILE")

if [ -z "$PACKAGES_TO_INSTALL" ]; then
    echo "YAMLファイルにインストール対象のパッケージが見つかりませんでした。"
    exit 0
fi

# APTキャッシュの更新とアップグレード
echo "APTキャッシュを更新しています..."
sudo apt update

echo "インストール済みパッケージをアップグレードしています..."
sudo apt upgrade -y

# パッケージのインストール
echo "パッケージをインストールします..."
# xargs を使って、改行区切りのリストをapt installコマンドに渡す
echo "$PACKAGES_TO_INSTALL" | xargs sudo apt install -y

echo -e "\n指定された全てのパッケージのインストールが完了しました"