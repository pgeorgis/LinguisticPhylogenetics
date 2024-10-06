#!/usr/bin/env bash
set -eou pipefail

# Create the virtual environment:
python3 -m venv venv

if declare -p GITHUB_USER_NAME >&/dev/null && declare -p GITHUB_ACCESS_TOKEN >&/dev/null; then
    CREDENTIALS_FILE="$HOME/.git-credentials"
    git config --global --replace-all credential.helper store
    touch "$CREDENTIALS_FILE"
    chmod 0600 "$CREDENTIALS_FILE"
cat <<EOF >> "$CREDENTIALS_FILE"
[credential "https://github.com"]
    username = $GITHUB_USER_NAME
    password = $GITHUB_ACCESS_TOKEN
EOF
fi

git submodule init && \
    git submodule update

# Activate virtual environment, install packages, and update submodules:
source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r requirements.txt
