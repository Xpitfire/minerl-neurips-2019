#!/bin/bash

PASSWD_PATH="${HOME}/.vnc/passwd"
XSTARTUP_PATH="${HOME}/.vnc/xstartup"
VNCSERVER="tigervncserver"
VNCPASSWD="tigervncpasswd"

vncserver_stop() {
    ${VNCSERVER} -clean -kill ${DISPLAY}
}

vncserver_start() {
    echo "${VNC_PW}" | ${VNCPASSWD} -f > ${PASSWD_PATH}
    chmod 600 ${PASSWD_PATH}
    ${VNCSERVER} ${DISPLAY} -localhost no
}

case "$1" in
    start)
        vncserver_start
    ;;

    stop)
        vncserver_stop
    ;;

    restart)
        vncserver_stop
        vncserver_start
    ;;

    *)
        echo "Usage: $0 <start|stop|restart>"
        exit 1
esac