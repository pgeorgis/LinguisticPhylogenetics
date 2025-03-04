#!/usr/bin/env bash
set -eou pipefail

# Initialize and update submodules
make sync-submodules

# Install R dependencies
./install_r_dependencies.R

# Create the virtual environment:
python3.12 -m venv venv || python3 -m venv venv

# Activate virtual environment, install packages
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
