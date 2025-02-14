#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 inputdir"
  exit 1
fi

inputdir="$1"

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# NB: these backends are resource-intensive to train
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#poetry run annif train wiki-svc-en $inputdir

# ensemble backends
poetry run annif train wiki-pav-en $inputdir
poetry run annif train wiki-nn-en $inputdir

# pecos / transformer
poetry run annif train wiki-xtransformer-en $inputdir