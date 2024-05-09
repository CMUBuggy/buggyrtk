#!/usr/bin/env bash

for i in {1..10}
do
  URL=$(docker compose logs | grep -m 1 -o http://127.0.0.1.*)
  if [ -z $URL ]; then
    echo Waiting for Jupyter server to start...
  else
    echo URL: $URL
    python -m webbrowser $URL
    # python is the most cross platform way to open a URL, but you could also use the following
    # Windows: start $URL
    # macOS: open $URL
    # Linux: xdg-open $URL
    break
  fi
  sleep 1
done
