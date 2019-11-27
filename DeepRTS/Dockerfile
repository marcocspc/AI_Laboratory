FROM ubuntu:18.04
MAINTAINER marcocspc 
ARG video

RUN if [ "$video" = "1" ] ; then echo '\033[0;32mVideo enabled, please remember to forward X session when running DeepRTS.\033[0m' ; else echo '\033[0;32mDeepRTS video disabled\033[0m' ; fi

ENV SDL_VIDEODRIVER=${video:+x11}
ENV SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}

RUN echo $SDL_VIDEODRIVER

RUN apt-get update && apt-get install -y apt-utils python3 python3-pip git xvfb build-essential cmake

RUN git clone https://github.com/UIA-CAIR/DeepRTS.git drts --recurse-submodules && cd drts && git checkout e54dc6c 
RUN pip3 install -e drts

RUN cat drts/coding/requirements.txt | xargs -n 1 pip3 install; exit 0
RUN cat drts/requirements.txt | xargs -n 1 pip3 install; exit 0

RUN cd && git clone https://github.com/marcocspc/AI_Laboratory.git
RUN cd && git clone https://github.com/pvnetto/URNAI-Tools.git 

RUN Xvfb :99 -ac &

ENTRYPOINT /bin/bash
