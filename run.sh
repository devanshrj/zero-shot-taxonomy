#!/bin/bash

# inputs
# $1: method to be evaluated
# $2: model checkpoint to be used
# $3: domain
# $4: prompt
METHOD=$1
MODEL=$2
DOMAIN=$3
PROMPT=$4

# generate taxonomies
echo "Generating taxonomies for method = $METHOD and model checkpoint = $MODEL..."
python3 models/$METHOD.py --model-name $MODEL --domain $DOMAIN --prompt $PROMPT
echo
echo "Taxonomies generated!"
echo
echo "Starting evaluation process for method = $METHOD and model checkpoint = $MODEL..."
python eval.py --method-name $METHOD --model-name $MODEL --domain $DOMAIN --prompt $PROMPT
echo
echo "Script finished."