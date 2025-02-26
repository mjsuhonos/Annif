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
poetry run annif train P361-mllm-en $inputdir
poetry run annif train P361-stwfsa-en $inputdir

# pecos / transformer
poetry run annif train P361-tfidf-en $inputdir
poetry run annif train P361-xtransformer-en $inputdir

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#warning: Unknown subject URI <http://www.wikidata.org/entity/Q16641459>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17347560>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1753637>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q40193>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q4030181>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q4164230>