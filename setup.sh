#!/usr/bin/env bash
set -eou pipefail

# Initialize and update submodules
git submodule update --init --recursive

# Create the virtual environment:
python3 -m venv venv

# Activate virtual environment, install packages
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
