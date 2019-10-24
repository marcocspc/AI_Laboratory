#This needs Docker 19.03 installed and nVidia Container Toolkit installed
#Install Docker-CE 19.03: https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#install-from-a-package 
#Install nVidia Container Toolkit: https://github.com/NVIDIA/nvidia-docker 

FROM nvidia/cuda:9.0-base
MAINTAINER marcocspc

RUN apt update && apt install git python3 python3-pip wget unzip -y
RUN cd ~ && git clone https://github.com/pvnetto/URNAI-Tools/
RUN pip3 install gym tensorflow-gpu pysc2 numpy matplotlib
RUN cd ~ && wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && unzip SC2.4.10.zip && rm SC2.4.10.zip
