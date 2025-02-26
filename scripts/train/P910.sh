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
poetry run annif train P910-mllm-en $inputdir
poetry run annif train P910-stwfsa-en $inputdir

# pecos / transformer
poetry run annif train P910-tfidf-en $inputdir
poetry run annif train P910-xtransformer-en $inputdir

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
