# FROM ubuntu:20.04
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04


# Set the working directory to /app
WORKDIR /app
RUN apt update 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y  python3-opencv libopencv-dev python3-pip git cmake
RUN pip --default-timeout=1000 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

RUN git clone https://github.com/tawat-ki/DAVAR-Lab-OCR.git
WORKDIR /app/DAVAR-Lab-OCR
RUN git checkout docker
RUN bash setup.sh
RUN pip install "numpy>=1.20"

