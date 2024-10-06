#!/usr/bin/env bash
set -eou pipefail

# Create the virtual environment:
python3 -m venv venv

if declare -p GITHUB_USER_NAME >&/dev/null && declare -p GITHUB_ACCESS_TOKEN >&/dev/null; then
    git config --global credential.helper store
    git credential approve <<EOF
protocol=https
host=github.com
username=${GITHUB_USER_NAME}
password=${GITHUB_ACCESS_TOKEN}
EOF
fi

git submodule update --init --recursive

# Activate virtual environment, install packages, and update submodules:
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
