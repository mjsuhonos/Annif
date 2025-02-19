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
poetry run annif train P279-tfidf-en $inputdir
poetry run annif train P279-mllm-en $inputdir
#poetry run annif train P279-stwfsa-en $inputdir

# pecos / transformer
poetry run annif train wiki-xtransformer-en $inputdir