FROM ubuntu:18.04
MAINTAINER Per-Arne Andersen

RUN apt-get update && apt-get install -y apt-utils python3 python3-pip git xvfb cmake 

RUN pip3 install --upgrade pip

RUN git clone https://github.com/cair/deep-rts.git && cd deep-rts && git checkout cace3c2c55 && git submodule update --init --recursive && pip3 install . 

RUN git clone https://github.com/marcocspc/AI_Laboratory.git

RUN Xvfb :99 -ac &

ENV SDL_VIDEODRIVER dummy

RUN python3 ./AI_Laboratory/DeepRTS/deep-rts.py
