#!/bin/bash

sudo apt-get update && sudo apt-get install -y apt-utils python3 python3-pip git xvfb build-essential cmake

git clone https://github.com/marcocspc/deep-rts drts --recurse-submodules && cd drts && git checkout stable && cd .. 
pip3 install -e drts

cat drts/coding/requirements.txt | xargs -n 1 pip3 install; exit 0
cat drts/requirements.txt | xargs -n 1 pip3 install; exit 0

