#!/usr/bin/env bash

set -euxo pipefail

docker compose build
docker compose up -d

docker exec -it dfp_dev bash