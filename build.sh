#!/usr/bin/env bash
set -e

echo "=== Cloning Kronos source code ==="
git clone https://github.com/shiyu-coder/Kronos.git /opt/kronos_src
cp -r /opt/kronos_src/model ./model

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Build complete ==="
