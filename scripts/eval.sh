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
poetry run annif eval wiki-tfidf-en $inputdir # zero scores on wiki5m
poetry run annif eval wiki-mllm-en $inputdir
#	MLLM
#Precision (doc avg):          	0.3101
#Recall (doc avg):             	0.7793
#F1 score (doc avg):           	0.4041
#Precision (subj avg):         	0.0866
#Recall (subj avg):            	0.1164
#F1 score (subj avg):          	0.0936
#Precision (weighted subj avg):	0.5801
#Recall (weighted subj avg):   	0.7793
#F1 score (weighted subj avg): 	0.6272
#Precision (microavg):         	0.2210
#Recall (microavg):            	0.7793
#F1 score (microavg):          	0.3444
#F1@5:                         	0.4206
#NDCG:                         	0.7449
#NDCG@5:                       	0.7431
#NDCG@10:                      	0.7449
#Precision@1:                  	0.7000
#Precision@3:                  	0.3607
#Precision@5:                  	0.3213
#True positives:               	1169
#False positives:              	4120
#False negatives:              	331
#Documents evaluated:          	1500

poetry run annif eval wiki-stwfsa-en $inputdir
#	STWFSA
#Precision (doc avg):          	0.1101
#Recall (doc avg):             	0.8073
#F1 score (doc avg):           	0.1887
#Precision (subj avg):         	0.0893
#Recall (subj avg):            	0.1205
#F1 score (subj avg):          	0.0964
#Precision (weighted subj avg):	0.5984
#Recall (weighted subj avg):   	0.8073
#F1 score (weighted subj avg): 	0.6457
#Precision (microavg):         	0.0985
#Recall (microavg):            	0.8073
#F1 score (microavg):          	0.1756
#F1@5:                         	0.2788
#NDCG:                         	0.7904
#NDCG@5:                       	0.7900
#NDCG@10:                      	0.7904
#Precision@1:                  	0.7673
#Precision@3:                  	0.2703
#Precision@5:                  	0.1701
#True positives:               	1211
#False positives:              	11080
#False negatives:              	289
#Documents evaluated:          	1500

poetry run annif eval wiki-yake-en $inputdir

# associative backends
#poetry run annif eval wiki-fasttext-en $inputdir # zero scores on wiki5m
#poetry run annif eval wiki-svc-en $inputdir # zero scores on wiki5m
# parabel
# bonsai

# ensemble backends
#poetry run annif eval wiki-ensemble-a-en $inputdir
poetry run annif eval wiki-ensemble-l-en $inputdir
#	Lexical Ensemble
#Precision (doc avg):          	0.0880
#Recall (doc avg):             	0.8800
#F1 score (doc avg):           	0.1600
#Precision (subj avg):         	0.0099
#Recall (subj avg):            	0.0131
#F1 score (subj avg):          	0.0108
#Precision (weighted subj avg):	0.6620
#Recall (weighted subj avg):   	0.8800
#F1 score (weighted subj avg): 	0.7206
#Precision (microavg):         	0.0880
#Recall (microavg):            	0.8800
#F1 score (microavg):          	0.1600
#F1@5:                         	0.2822
#NDCG:                         	0.7976
#NDCG@5:                       	0.7868
#NDCG@10:                      	0.7976
#Precision@1:                  	0.7000
#Precision@3:                  	0.2756
#Precision@5:                  	0.1693
#True positives:               	132
#False positives:              	1368
#False negatives:              	18
#Documents evaluated:          	150

poetry run annif eval wiki-pav-en $inputdir
#	PAV
#Precision (doc avg):          	0.0763
#Recall (doc avg):             	0.7633
#F1 score (doc avg):           	0.1388
#Precision (subj avg):         	0.0850
#Recall (subj avg):            	0.1140
#F1 score (subj avg):          	0.0912
#Precision (weighted subj avg):	0.5695
#Recall (weighted subj avg):   	0.7633
#F1 score (weighted subj avg): 	0.6108
#Precision (microavg):         	0.0763
#Recall (microavg):            	0.7633
#F1 score (microavg):          	0.1388
#F1@5:                         	0.2398
#NDCG:                         	0.6755
#NDCG@5:                       	0.6612
#NDCG@10:                      	0.6755
#Precision@1:                  	0.5927
#Precision@3:                  	0.2267
#Precision@5:                  	0.1439
#True positives:               	1145
#False positives:              	13855
#False negatives:              	355
#Documents evaluated:          	1500

poetry run annif eval wiki-nn-en $inputdir
#	NN
#Precision (doc avg):          	0.3561
#Recall (doc avg):             	0.8713
#F1 score (doc avg):           	0.4549
#Precision (subj avg):         	0.1083
#Recall (subj avg):            	0.1301
#F1 score (subj avg):          	0.1138
#Precision (weighted subj avg):	0.7255
#Recall (weighted subj avg):   	0.8713
#F1 score (weighted subj avg): 	0.7620
#Precision (microavg):         	0.2154
#Recall (microavg):            	0.8713
#F1 score (microavg):          	0.3454
#F1@5:                         	0.4743
#NDCG:                         	0.8150
#NDCG@5:                       	0.8117
#NDCG@10:                      	0.8150
#Precision@1:                  	0.7420
#Precision@3:                  	0.4144
#Precision@5:                  	0.3694
#True positives:               	1307
#False positives:              	4761
#False negatives:              	193
#Documents evaluated:          	1500

# pecos / transformer
poetry run annif eval wiki-xtransformer-en $inputdir # zero scores on wiki5m
