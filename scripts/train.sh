#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 inputdir"
  exit 1
fi

inputdir="$1"

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# train lightweight backends for assessment
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# lexical backends
poetry run annif train wiki-tfidf-en $inputdir
poetry run annif train wiki-mllm-en $inputdir
poetry run annif train wiki-stwfsa-en $inputdir

# associative backends
poetry run annif train wiki-fasttext-en $inputdir

# only available on macos (arm) or linux (x64)
# ie. linux aarch64 not supported

#poetry run annif train wiki-parabel-en $inputdir
#poetry run annif train wiki-bonsai-en $inputdir

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# NB: these backends are resource-intensive to train
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

poetry run annif train wiki-svc-en $inputdir

# ensemble backends
poetry run annif train wiki-pav-en $inputdir
poetry run annif train wiki-nn-en $inputdir

# pecos / transformer
poetry run annif train wiki-xtransformer-en $inputdir