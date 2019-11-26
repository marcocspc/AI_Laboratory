FROM ubuntu:18.04
MAINTAINER Per-Arne Andersen

RUN apt-get update && apt-get install -y apt-utils python3 python3-pip git xvfb build-essential cmake

RUN git clone https://github.com/UIA-CAIR/DeepRTS.git drts --recurse-submodules && cd deep-rts && git checkout c57a168 
RUN pip3 install -e drts

RUN cat drts/coding/requirements.txt | xargs -n 1 pip3 install; exit 0
RUN cat drts/requirements.txt | xargs -n 1 pip3 install; exit 0

RUN git clone https://github.com/marcocspc/AI_Laboratory.git

RUN Xvfb :99 -ac &

ENV SDL_VIDEODRIVER dummy

RUN python3 ./AI_Laboratory/DeepRTS/deep-rts.py
