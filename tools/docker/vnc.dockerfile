FROM ml-jku/base:latest

LABEL MAINTAINER="gfrogat@gmail.com"
ENV REFRESHED_AT 2019-06-09

# Switch to root for installing everything
USER 0

## UI Connections
ENV DISPLAY=:1 \
    VNC_PORT=5901 \
    STARTUPDIR=/dockerstartup \
    DEBIAN_FRONTEND=noninteractive
EXPOSE ${VNC_PORT} 

RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    xfonts-base \
    tigervnc-standalone-server \
    tigervnc-xorg-extension && \
    rm -rf /var/lib/apt/lists/*

# Copy default xfce4 config
COPY resources/config/xfce4 /home/ml/.config/xfce4

# Copy default vnc config
COPY resources/vnc /home/ml/.vnc/
RUN chown -R ml:ml /home/ml/.vnc && \
    chmod +x /home/ml/.vnc/xstartup

# Copy startup scripts
COPY resources/scripts ${STARTUPDIR}
RUN chmod +x ${STARTUPDIR}/*

# Switch back to default user
USER ml 
WORKDIR /home/ml
