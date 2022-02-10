#!/bin/sh
sudo apt-get install python3.8-dev python3.8-venv
python3.8 -m venv lava_env
source lava_env/bin/activate
pip install -U pip
cd lava
pip install -r build-requirements.txt
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src
pyb -E unit
cd ..
cd lava-dl
pip install -r build-requirements.txt
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pyb -E unit
cd ..
cd feature_extract
# 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