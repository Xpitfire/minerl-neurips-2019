#!/bin/bash

echo "Starting VNC Servec"
/dockerstartup/start_vnc.sh start

echo "Starting JupyterLab"
jupyter lab --ip 0.0.0.0 --no-browser
