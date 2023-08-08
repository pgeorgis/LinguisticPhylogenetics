# Create the virtual environment:
python3 -m venv venv

# Activate virtual environment:
source venv/bin/activate

# Install required packages:
venv/bin/pip3 install -r requirements.txt

# Install submodules:
git submodule init

git submodule update