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
poetry run annif train wiki-parabel-en $inputdir
poetry run annif train wiki-bonsai-en $inputdir