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
#poetry run annif eval P361-tfidf-en $inputdir
# zero scores on wiki5m

#poetry run annif eval P361-mllm-en $inputdir
#poetry run annif eval P361-stwfsa-en $inputdir

# ensemble backends
poetry run annif eval P361-ensemble-l-en $inputdir
poetry run annif eval P361-ensemble-alt-en $inputdir

#poetry run annif eval P361-yake-en $inputdir

# pecos / transformer
#poetry run annif eval P361-xtransformer-en $inputdir
# zero scores on wiki5m

#poetry run annif eval P361-pav-en $inputdir
