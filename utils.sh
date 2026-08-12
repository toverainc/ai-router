#!/bin/bash

set -e

AIROUTER_CONFIG_FILE="${AIROUTER_CONFIG_FILE:-ai-router.toml}"
PORT=${PORT:-3000}

case $1 in

build)
    docker build -t ai-router .
;;

run)
    docker run --rm -it -p "$PORT":3000 \
	    --volume="${PWD}/${AIROUTER_CONFIG_FILE}:/etc/ai-router/config.toml" \
	    ai-router:latest
;;

run-local)
    docker run --rm -it --net=host \
	    --volume="${PWD}/${AIROUTER_CONFIG_FILE}:/etc/ai-router/config.toml" \
	    ai-router:latest
;;

esac
