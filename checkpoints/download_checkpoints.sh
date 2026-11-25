#!/usr/bin/env bash
# Download the three released models into this folder: omnix/, jets_only/ and events_only/,
# each with config.yaml and model.ckpt (1.4 GB in total).
#
#   ./checkpoints/download_checkpoints.sh
set -euo pipefail

URL="https://syncandshare.desy.de/public.php/dav/files/MzYEECCJxYzyjcs/omnix_checkpoints.tar.gz"
SHA256="2eaf8ffeb8740d6931c0e97c332158597244b2ca1c2c97703b0c8267248ed5a3"

cd "$(dirname "$0")"
curl --location --fail --output omnix_checkpoints.tar.gz "$URL"
echo "$SHA256  omnix_checkpoints.tar.gz" | sha256sum --check
tar -xzf omnix_checkpoints.tar.gz
rm omnix_checkpoints.tar.gz
ls -d */
