FROM nvidia/opengl:1.1-glvnd-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

RUN apt-get -qq update 
RUN apt-get -qq install -y sudo curl gconf-service
RUN apt-get -qq install -y lib32gcc1 lib32stdc++6 libasound2 libc6 libc6-i386 
RUN apt-get -qq install -y libcairo2 libcap2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libfreetype6 libgcc1 libgconf-2-4 
RUN apt-get -qq install -y libgdk-pixbuf2.0-0 libgl1-mesa-glx mesa-utils libglib2.0-0 libglu1-mesa libgtk2.0-0 libnspr4 libnss3 libpango1.0-0 libstdc++6 libx11-6 libxcomposite1 libxcursor1 libxdamage1 libxext6 
RUN apt-get -qq install -y libxfixes3 libxi6 libxrandr2 libxrender1 libxtst6 zlib1g debconf npm xdg-utils lsb-release libpq5 xvfb x11-apps && rm -rf /var/lib/apt/lists/*
RUN apt-get -qq update
RUN apt-get -qq install -y --no-install-recommends wget curl unzip
RUN apt-get -qq install -y python3.8 python3.8-tk virtualenv

COPY find_fallen_challenge /find_fallen_challenge
COPY multimodal_challenge /multimodal_challenge
COPY split /split
COPY TDW /TDW
COPY interface.py /interface.py
COPY requirements.txt /requirements.txt

RUN virtualenv venv -p python3.8
RUN venv/bin/pip install -r requirements.txt
