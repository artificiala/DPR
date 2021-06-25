FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN apt-get update && apt-get install -y \
  htop \
  screen

WORKDIR /workspace

RUN git clone https://github.com/artificiala/DPR.git \
    && cd DPR \
    && pip install -e .

RUN pip install wandb tensorboard sentencepiece swifter jsonlines pandas