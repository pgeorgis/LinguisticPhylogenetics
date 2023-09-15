# Create the virtual environment:
python3 -m venv venv

# Activate virtual environment, install packages, and update submodules:
source venv/bin/activate && \
    pip3 install -r requirements.txt && \
    git submodule init && \
    git submodule update