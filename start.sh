#!/usr/bin/env bash

docker compose build
docker compose up -d

./open-jupyter.sh
