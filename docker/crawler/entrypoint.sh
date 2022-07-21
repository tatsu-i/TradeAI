#!/bin/bash
set -o pipefail

# crond start
busybox crond -f -L /dev/stder
