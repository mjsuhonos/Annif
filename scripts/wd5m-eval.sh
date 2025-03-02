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

#poetry run annif eval P279-mllm-en $inputdir > P279-mllm-en.txt
#poetry run annif eval P279-stwfsa-en $inputdir > P279-stwfsa-en.txt

#poetry run annif eval P361-mllm-en $inputdir > P361-mllm-en.txt
#poetry run annif eval P361-stwfsa-en $inputdir > P361-stwfsa-en.txt

#poetry run annif eval P910-mllm-en $inputdir > P910-mllm-en.txt
#poetry run annif eval P910-stwfsa-en $inputdir > P910-stwfsa-en.txt

poetry run annif eval uP279_P910_P361-mllm-en $inputdir > uP279_P910_P361-mllm-en.txt

poetry run annif eval uP279_P910_P361-stwfsa-en $inputdir > uP279_P910_P361-stwfsa-en.txt

# 10GB (? to confirm)
poetry run annif eval ensemble-2-en $inputdir > ensemble-2-en.txt

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# WARNING: > 30GB RAM required
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
poetry run annif eval ensemble-3-en $inputdir > ensemble-3-en.txt
poetry run annif eval ensemble-4-en $inputdir > ensemble-4-en.txt
