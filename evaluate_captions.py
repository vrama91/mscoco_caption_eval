#!/usr/bin/env python
# 
# File Name : evaluate_captions.py
#
# Description :
#
# Usage :
#
# Creation Date : 06-01-2015
# Last Modified : Tue Jan  6 21:47:15 2015
# Author : Hao Fang

import os

path_to_tokenized_ref_json = 'data/tokenized_ref.json'
path_to_raw_hypo_json = 'data/hypo.json'

from caption_eval.evals import PTBTokenizer, Bleu, Rouge, Meteor, Cider

# =================================================
# Load references
# =================================================
import json
tokenized_ref_for_image = json.load(open(path_to_tokenized_ref_json))

# =================================================
# Tokenize hypotheses
# =================================================
print 'tokenizing hypothese...'
raw_hypo_for_image = json.load(open(path_to_raw_hypo_json))
tokenizer = PTBTokenizer()
tokenized_hypo_for_image= tokenizer.tokenize(raw_hypo_for_image)
keys1 = tokenized_hypo_for_image.keys() 
keys1.sort()
keys2 = tokenized_ref_for_image.keys()
keys2.sort()
assert(keys1 == keys2)
print 'all hypothese have been tokenized'

# =================================================
# Compute Bleu (up to 4-gram)
# - using 'shortest' reference length as brevity penalty
# Also provides n-gram precision w/o brevity penalty (This is NOT Bleu)
# =================================================
scorer = Bleu()
score, bleu_info = scorer.compute_score(tokenized_hypo_for_image, \
        tokenized_ref_for_image)
print 'Bleu: ', score
print 'Bleu info: ', bleu_info

# =================================================
# Compute Meteor
# =================================================
scorer = Meteor()
score = scorer.compute_score(tokenized_hypo_for_image, \
        tokenized_ref_for_image)
print 'Meteor: ', score

#...
