FROM ml-jku/vnc

LABEL MAINTAINER="gfrogat@gmail.com"
ENV REFRESHED_AT 2019-06-09

LABEL com.nvidia.volumes.needed="nvidia_driver"
USER 0

ARG TORCH_WHEEL=https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl 
ARG TORCHVISION_WHEEL=https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    python-tk \
    python-imaging-tk \
    openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/*

# Note the trailing slash - essential!
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> /home/ml/.bashrc

# Set CUDA related environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Update pip 
RUN sudo pip3 install -U pip setuptools
RUN sudo pip3 install futures pillow

# Install PyTorch
RUN sudo pip3 install ${TORCH_WHEEL}
RUN sudo pip3 install ${TORCHVISION_WHEEL}

# Install Jupyter, ptvsd 
RUN sudo pip3 install --no-cache jupyterlab
RUN sudo pip3 install --no-cache ptvsd

# Switch back to default user
USER ml 
WORKDIR /home/ml

# Install MineRL for user due to Gradle Errors
RUN pip3 install --no-cache --user minerl 

ENTRYPOINT ["/dockerstartup/console_startup.sh"]