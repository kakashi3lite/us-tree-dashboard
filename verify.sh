#!/usr/bin/env bash
set -e
pip install -r requirements.txt
pytest --maxfail=1 --disable-warnings -q
docker-compose build --quiet
docker-compose up --abort-on-container-exit
