#!/usr/bin/env bash
set -e

echo "[1/6] System packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git build-essential

echo "[2/6] Create venv"
python3 -m venv .venv
source .venv/bin/activate

echo "[3/6] Upgrade pip tooling"
python -m pip install --upgrade pip wheel setuptools

echo "[4/6] Install (Gemini minimal) with constraints"
pip install -c env/constraints.txt -r env/requirements.txt

echo "[5/6] Sanity import test"
python -c "import langchain, langsmith, faiss, numpy, pandas; print('OK: imports')"

echo "[6/6] Done"
echo "Activate next time: source .venv/bin/activate"
