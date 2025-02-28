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
poetry run annif train uP279_P910_P361-mllm-en $inputdir
poetry run annif train uP279_P910_P361-stwfsa-en $inputdir

# pecos / transformer
poetry run annif train uP279_P910_P361-tfidf-en $inputdir
poetry run annif train uP279_P910_P361-xtransformer-en $inputdir

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#warning: Unknown subject URI <http://www.wikidata.org/entity/Q10816>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1207406>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1226512>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q130360>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q13567091>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1391360>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1427700>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q15818161>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1636342>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q16641459>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17009282>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17347560>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17405077>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1753637>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q17580276>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1797895>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q18204558>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q1981640>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2306215>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2420931>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2474839>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q2512021>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q3065672>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q3433211>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q3788714>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q40193>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q4030181>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q4164230>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q513420>
