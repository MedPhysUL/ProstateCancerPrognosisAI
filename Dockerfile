ARG CUDA_VERSION=11.7.0
ARG UBUNTU_VERSION=22.04
ARG CUDNN_VERSION=8

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

# To avoid tzdata errors
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get -y clean && \
    apt-get -y update && \
    ln -fs /usr/share/zoneinfo/America/Toronto /etc/localtime && \
    apt-get -y --no-install-recommends install build-essential wget libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev zlib1g-dev libncurses5-dev libncursesw5-dev python3-opencv  && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Download Python 3.11.2 source code and extract it
RUN wget https://www.python.org/ftp/python/3.11.2/Python-3.11.2.tgz && \
    tar -xvf Python-3.11.2.tgz && \
    cd Python-3.11.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm Python-3.11.2.tgz

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py

RUN python3.11 --version

RUN pip3.11 --version

RUN mkdir /workspace

RUN mkdir /workspace/src

COPY src /workspace/src

COPY requirements.txt /workspace

RUN pip3 install -r /workspace/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

RUN python3.11 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

RUN mkdir /workspace/applications

COPY applications/constants.py /workspace/applications

COPY applications/env_apps.py /workspace/applications

COPY applications/logging_conf.yaml /workspace/applications

COPY applications/main.py /workspace/applications

RUN mkdir /workspace/applications/local_data

COPY applications/local_data/learning_set.h5 /workspace/applications/local_data

COPY applications/local_data/learning_table.csv /workspace/applications/local_data

RUN mkdir /workspace/applications/local_data/masks

COPY applications/local_data/masks /workspace/applications/local_data/masks

# Create launcher
RUN echo "#!/bin/bash" >> /workspace/run_tuner.sh
RUN echo "cd /workspace/applications" >> /workspace/run_tuner.sh
RUN echo "python3.11 \${1}" >> /workspace/run_tuner.sh
RUN chmod a+x /workspace/run_tuner.sh

ENTRYPOINT ["/workspace/run_tuner.sh"]
