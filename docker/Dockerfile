FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get -y clean && \
    apt-get -y update && \
    ln -fs /usr/share/zoneinfo/America/Toronto /etc/localtime && \
    apt-get -y --no-install-recommends install python3-opencv  && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN python3 --version

RUN pip3 --version

RUN python3 -c "import torch; print(torch.__version__)"

RUN pip3 install numpy==1.24.2

RUN mkdir /workspace/src

COPY src /workspace/src

COPY requirements_docker.txt /workspace

RUN pip3 install -r /workspace/requirements_docker.txt

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
RUN echo "python3 \${1}" >> /workspace/run_tuner.sh
RUN chmod a+x /workspace/run_tuner.sh

ENTRYPOINT ["/workspace/run_tuner.sh"]
