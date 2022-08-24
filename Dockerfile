FROM python:3.10
RUN apt update && apt install -y python3-pip
RUN /usr/local/bin/python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
WORKDIR /biip
RUN mkdir biip && mkdir data
ADD biip biip/ 
ADD data data/
ADD generate_data.py inference.py train.py ./
RUN mkdir artifacts
RUN mkdir artifacts/experiment_0
RUN mkdir artifacts/experiment_0/inference
RUN mkdir artifacts/experiment_0/model
RUN mkdir artifacts/experiment_0/train
RUN python train.py
RUN python inference.py
