#!/bin/bash
websockify --web=/usr/share/novnc/  6080 ${VNC_HOST}:${VNC_PORT}
