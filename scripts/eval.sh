#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 inputdir"
  exit 1
fi

inputdir="$1"

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# eval lightweight backends for assessment
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# lexical backends
poetry run annif eval wiki-tfidf-en $inputdir # zero scores on wiki5m
poetry run annif eval wiki-mllm-en $inputdir
poetry run annif eval wiki-stwfsa-en $inputdir
poetry run annif eval wiki-yake-en $inputdir

# associative backends
#poetry run annif eval wiki-fasttext-en $inputdir # zero scores on wiki5m
#poetry run annif eval wiki-svc-en $inputdir # zero scores on wiki5m
# parabel
# bonsai

# ensemble backends
#poetry run annif eval wiki-ensemble-a-en $inputdir
poetry run annif eval wiki-ensemble-l-en $inputdir
poetry run annif eval wiki-pav-en $inputdir
poetry run annif eval wiki-nn-en $inputdir

# pecos / transformer
poetry run annif eval wiki-xtransformer-en $inputdir # zero scores on wiki5m