FROM ubuntu:18.04

LABEL MAINTAINER="gfrogat@gmail.com"
ENV REFRESHED_AT 2019-06-09

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libjpeg-dev \
    libpng-dev \
    libpython3.6 \
    python3-pip \
    python3-dev \
    sudo \
    unzip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN sudo update-ca-certificates -f

# Install and update locale
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales && \
    rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'

# Create user ml
RUN useradd --create-home --shell /bin/bash --no-log-init --groups sudo ml 
RUN sudo bash -c 'echo "ml ALL=(ALL:ALL) NOPASSWD: ALL" | (EDITOR="tee -a" visudo)'

USER ml
WORKDIR /home/ml
