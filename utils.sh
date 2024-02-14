#!/bin/bash

set -e

PORT=3000

case $1 in

build)
    docker build -t ai-router .
;;

run)
    docker run --rm -it -p "$PORT":3000 ai-router:latest
;;

run-local)
    docker run --rm -it --net=host ai-router:latest
;;

esac
