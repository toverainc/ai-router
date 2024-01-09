#!/bin/bash

set -e

PORT=3000
TRITON_ENDPOINT="http://localhost:8001"

case $1 in

build)
    docker build -t ai-router .
;;

run)
    docker run --rm -it -p "$PORT":3000 ai-router:latest \
        --host 0.0.0.0 --port 3000 --triton-endpoint "$TRITON_ENDPOINT"
;;

run-local)
    docker run --rm -it --net=host ai-router:latest \
        --host 0.0.0.0 --port "$PORT" --triton-endpoint "$TRITON_ENDPOINT"
;;

esac