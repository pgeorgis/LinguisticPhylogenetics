#!/usr/bin/env bash
set -eou pipefail

# Create the virtual environment:
python3 -m venv venv

if [ -z "$GITHUB_ACCESS_TOKEN" ] && [ -z "$GITHUB_USER_NAME" ]; then
    touch "$HOME"/.git-credentials
    chmod 0600 "$HOME"/.git-credentials
cat <<EOF >> "$HOME/.git-credentials"
[credential]
    username = $GITHUB_USER_NAME
    password = $GITHUB_ACCESS_TOKEN
EOF
    git config --global credential.helper store
fi

# Activate virtual environment, install packages, and update submodules:
source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r requirements.txt && \
    git submodule init && \
    git submodule update
