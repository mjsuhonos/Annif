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
#poetry run annif eval P910-tfidf-en $inputdir
# zero scores on wiki5m

#poetry run annif eval P910-mllm-en $inputdir
#poetry run annif eval P910-stwfsa-en $inputdir

# ensemble backends
poetry run annif eval P910-ensemble-l-en $inputdir
poetry run annif eval P910-ensemble-alt-en $inputdir

#poetry run annif eval P910-yake-en $inputdir

# pecos / transformer
#poetry run annif eval P910-xtransformer-en $inputdir
# zero scores on wiki5m

#poetry run annif eval P910-pav-en $inputdir
