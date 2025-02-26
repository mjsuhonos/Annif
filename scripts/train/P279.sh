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
poetry run annif train P279-mllm-en $inputdir
poetry run annif train P279-stwfsa-en $inputdir

# pecos / transformer
poetry run annif train P279-tfidf-en $inputdir
poetry run annif train P279-xtransformer-en $inputdir

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#warning: Unknown subject URI <http://www.wikidata.org/entity/Q10816>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1207406>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1226512>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q130360>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1391360>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1427700>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1636342>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1797895>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q18204558>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1981640>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2306215>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q3433211>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q513420>