FROM Ubuntu
MAINTAINER marcocspc

RUN apt update && apt install git python3 python3-pip wget unzip -y
RUN cd ~ && git clone https://github.com/pvnetto/URNAI-Tools/
RUN pip3 install gym tensorflow pysc2 numpy matplotlib
RUN cd ~ && wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && unzip SC2.4.10.zip && rm SC2.4.10.zip
