#!/bin/bash

set -e

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY

case $1 in

venv)
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
;;

*)
    source venv/bin/activate
    python "$@"
;;

esac