FROM ubuntu:18.04
MAINTAINER marcocspc 
ARG video

ENV DISPLAY=host.docker.internal:0

RUN apt-get update && apt-get install -y apt-utils python3 python3-pip git xvfb build-essential cmake

RUN git clone https://github.com/marcocspc/deep-rts drts --recurse-submodules && cd drts && git checkout stable && cd .. 
RUN pip3 install -e drts

RUN cat drts/coding/requirements.txt | xargs -n 1 pip3 install; exit 0
RUN cat drts/requirements.txt | xargs -n 1 pip3 install; exit 0

RUN rm -rf drts

#You should mount this folder as a volume to a folder where you are storing your scripts
#Something like docker run -it -v /path/on/hotst:/root/git --name <container-name> <image-name>
RUN mkdir /root/git

ENTRYPOINT /bin/bash
