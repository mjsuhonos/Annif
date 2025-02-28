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

#warning: Unknown subject URI <http://www.wikidata.org/entity/Q10816>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q13567091>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q15818161>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17009282>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17405077>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17580276>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2420931>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2474839>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2512021>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q3065672>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q3788714>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q4164230>
