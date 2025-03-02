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
poetry run annif eval P279-tfidf-en $inputdir
# zero scores on wiki5m

poetry run annif eval P279-mllm-en $inputdir
#Precision (doc avg):          	0.0879
#Recall (doc avg):             	0.8618
#F1 score (doc avg):           	0.1591
#Precision (subj avg):         	0.1017
#Recall (subj avg):            	0.1292
#F1 score (subj avg):          	0.1078
#Precision (weighted subj avg):	0.6782
#Recall (weighted subj avg):   	0.8619
#F1 score (weighted subj avg): 	0.7189
#Precision (microavg):         	0.0869
#Recall (microavg):            	0.8619
#F1 score (microavg):          	0.1579
#F1@5:                         	0.2824
#NDCG:                         	0.8048
#NDCG@5:                       	0.7988
#NDCG@10:                      	0.8048
#Precision@1:                  	0.7412
#Precision@3:                  	0.2758
#Precision@5:                  	0.1697
#True positives:               	25545
#False positives:              	268390
#False negatives:              	4093
#Documents evaluated:          	29643

# RE-RUN with full vocabulary 26/6/2025
#Precision (doc avg):          	0.0853
#Recall (doc avg):             	0.8478
#F1 score (doc avg):           	0.1548
#Precision (subj avg):         	0.0250
#Recall (subj avg):            	0.0304
#F1 score (subj avg):          	0.0263
#Precision (weighted subj avg):	0.6973
#Recall (weighted subj avg):   	0.8479
#F1 score (weighted subj avg): 	0.7343
#Precision (microavg):         	0.0849
#Recall (microavg):            	0.8479
#F1 score (microavg):          	0.1544
#F1@5:                         	0.2748
#NDCG:                         	0.7712
#NDCG@5:                       	0.7627
#NDCG@10:                      	0.7712
#Precision@1:                  	0.6895
#Precision@3:                  	0.2656
#Precision@5:                  	0.1650
#True positives:               	25130
#False positives:              	270841
#False negatives:              	4508
#Documents evaluated:          	29643


poetry run annif eval P279-stwfsa-en $inputdir
#Precision (doc avg):          	0.0860
#Recall (doc avg):             	0.7311
#F1 score (doc avg):           	0.1506
#Precision (subj avg):         	0.0961
#Recall (subj avg):            	0.1096
#F1 score (subj avg):          	0.0994
#Precision (weighted subj avg):	0.6414
#Recall (weighted subj avg):   	0.7312
#F1 score (weighted subj avg): 	0.6634
#Precision (microavg):         	0.0789
#Recall (microavg):            	0.7312
#F1 score (microavg):          	0.1425
#F1@5:                         	0.2484
#NDCG:                         	0.7157
#NDCG@5:                       	0.7150
#NDCG@10:                      	0.7157
#Precision@1:                  	0.6948
#Precision@3:                  	0.2435
#Precision@5:                  	0.1507
#True positives:               	21671
#False positives:              	252916
#False negatives:              	7967
#Documents evaluated:          	29643

# ensemble backends
poetry run annif eval P279-ensemble-l-en $inputdir

poetry run annif eval P279-ensemble-alt-en $inputdir

# sources=P279-tfidf-en,P279-mllm-en,P279-yake-en,P279-stwfsa-en
#Precision (doc avg):          	0.0825
#Recall (doc avg):             	0.8252
#F1 score (doc avg):           	0.1501
#Precision (subj avg):         	0.1115
#Recall (subj avg):            	0.1237
#F1 score (subj avg):          	0.1144
#Precision (weighted subj avg):	0.7438
#Recall (weighted subj avg):   	0.8253
#F1 score (weighted subj avg): 	0.7631
#Precision (microavg):         	0.0825
#Recall (microavg):            	0.8253
#F1 score (microavg):          	0.1500
#F1@5:                         	0.2622
#NDCG:                         	0.7469
#NDCG@5:                       	0.7343
#NDCG@10:                      	0.7469
#Precision@1:                  	0.6712
#Precision@3:                  	0.2521
#Precision@5:                  	0.1573
#True positives:               	24461
#False positives:              	271961
#False negatives:              	5177
#Documents evaluated:          	29643

# sources=P279-mllm-en,P279-yake-en,P279-stwfsa-en
#Precision (doc avg):          	0.0846
#Recall (doc avg):             	0.8319
#F1 score (doc avg):           	0.1532
#Precision (subj avg):         	0.1112
#Recall (subj avg):            	0.1247
#F1 score (subj avg):          	0.1144
#Precision (weighted subj avg):	0.7420
#Recall (weighted subj avg):   	0.8320
#F1 score (weighted subj avg): 	0.7628
#Precision (microavg):         	0.0837
#Recall (microavg):            	0.8320
#F1 score (microavg):          	0.1522
#F1@5:                         	0.2705
#NDCG:                         	0.7831
#NDCG@5:                       	0.7761
#NDCG@10:                      	0.7831
#Precision@1:                  	0.7322
#Precision@3:                  	0.2649
#Precision@5:                  	0.1624
#True positives:               	24659
#False positives:              	269788
#False negatives:              	4979
#Documents evaluated:          	29643

#sources=P279-mllm-en,P279-stwfsa-en,P279-tfidf-en,P279-xtransformer-en
#Precision (doc avg):          	0.0796
#Recall (doc avg):             	0.7957
#F1 score (doc avg):           	0.1447
#Precision (subj avg):         	0.1157
#Recall (subj avg):            	0.1193
#F1 score (subj avg):          	0.1168
#Precision (weighted subj avg):	0.7719
#Recall (weighted subj avg):   	0.7958
#F1 score (weighted subj avg): 	0.7795
#Precision (microavg):         	0.0796
#Recall (microavg):            	0.7958
#F1 score (microavg):          	0.1447
#F1@5:                         	0.2614
#NDCG:                         	0.7477
#NDCG@5:                       	0.7440
#NDCG@10:                      	0.7477
#Precision@1:                  	0.6881
#Precision@3:                  	0.2572
#Precision@5:                  	0.1568
#True positives:               	23586
#False positives:              	272844
#False negatives:              	6052
#Documents evaluated:          	29643

#sources=P279-tfidf-en,P279-xtransformer-en
#...

#sources=P279-mllm-en,P279-stwfsa-en,P279-xtransformer-en
#Precision (doc avg):          	0.0823
#Recall (doc avg):             	0.8232
#F1 score (doc avg):           	0.1497
#Precision (subj avg):         	0.1175
#Recall (subj avg):            	0.1234
#F1 score (subj avg):          	0.1194
#Precision (weighted subj avg):	0.7839
#Recall (weighted subj avg):   	0.8234
#F1 score (weighted subj avg): 	0.7962
#Precision (microavg):         	0.0823
#Recall (microavg):            	0.8234
#F1 score (microavg):          	0.1497
#F1@5:                         	0.2739
#NDCG:                         	0.8038
#NDCG@5:                       	0.8033
#NDCG@10:                      	0.8038
#Precision@1:                  	0.7752
#Precision@3:                  	0.2730
#Precision@5:                  	0.1644
#True positives:               	24403
#False positives:              	272027
#False negatives:              	5235
#Documents evaluated:          	29643

#sources=P279-mllm-en,P279-stwfsa-en,P279-tfidf-en
#Precision (doc avg):          	0.0822
#Recall (doc avg):             	0.8216
#F1 score (doc avg):           	0.1494
#Precision (subj avg):         	0.1175
#Recall (subj avg):            	0.1232
#F1 score (subj avg):          	0.1193
#Precision (weighted subj avg):	0.7838
#Recall (weighted subj avg):   	0.8218
#F1 score (weighted subj avg): 	0.7957
#Precision (microavg):         	0.0822
#Recall (microavg):            	0.8218
#F1 score (microavg):          	0.1494
#F1@5:                         	0.2720
#NDCG:                         	0.7908
#NDCG@5:                       	0.7889
#NDCG@10:                      	0.7908
#Precision@1:                  	0.7495
#Precision@3:                  	0.2700
#Precision@5:                  	0.1632
#True positives:               	24356
#False positives:              	272066
#False negatives:              	5282
#Documents evaluated:          	29643

#sources=P279-mllm-en,P279-stwfsa-en
#Precision (doc avg):          	0.0894
#Recall (doc avg):             	0.8805
#F1 score (doc avg):           	0.1620
#Precision (subj avg):         	0.1036
#Recall (subj avg):            	0.1320
#F1 score (subj avg):          	0.1099
#Precision (weighted subj avg):	0.6911
#Recall (weighted subj avg):   	0.8807
#F1 score (weighted subj avg): 	0.7332
#Precision (microavg):         	0.0886
#Recall (microavg):            	0.8807
#F1 score (microavg):          	0.1611
#F1@5:                         	0.2912
#NDCG:                         	0.8485
#NDCG@5:                       	0.8457
#NDCG@10:                      	0.8485
#Precision@1:                  	0.8099
#Precision@3:                  	0.2881
#Precision@5:                  	0.1749
#True positives:               	26102
#False positives:              	268338
#False negatives:              	3536
#Documents evaluated:          	29643

# RE-RUN with full vocabulary 26/6/2025
#Precision (doc avg):          	0.0871
#Recall (doc avg):             	0.8695
#F1 score (doc avg):           	0.1583
#Precision (subj avg):         	0.0256
#Recall (subj avg):            	0.0312
#F1 score (subj avg):          	0.0270
#Precision (weighted subj avg):	0.7150
#Recall (weighted subj avg):   	0.8697
#F1 score (weighted subj avg): 	0.7533
#Precision (microavg):         	0.0870
#Recall (microavg):            	0.8697
#F1 score (microavg):          	0.1582
#F1@5:                         	0.2847
#NDCG:                         	0.8167
#NDCG@5:                       	0.8116
#NDCG@10:                      	0.8167
#Precision@1:                  	0.7571
#Precision@3:                  	0.2790
#Precision@5:                  	0.1709
#True positives:               	25775
#False positives:              	270425
#False negatives:              	3863
#Documents evaluated:          	29643

#poetry run annif eval P279-yake-en $inputdir
#Precision (doc avg):          	0.0590
#Recall (doc avg):             	0.5599
#F1 score (doc avg):           	0.1060
#Precision (subj avg):         	0.0744
#Recall (subj avg):            	0.0839
#F1 score (subj avg):          	0.0765
#Precision (weighted subj avg):	0.4966
#Recall (weighted subj avg):   	0.5600
#F1 score (weighted subj avg): 	0.5102
#Precision (microavg):         	0.0573
#Recall (microavg):            	0.5600
#F1 score (microavg):          	0.1040
#F1@5:                         	0.1522
#NDCG:                         	0.3707
#NDCG@5:                       	0.3369
#NDCG@10:                      	0.3707
#Precision@1:                  	0.2090
#Precision@3:                  	0.1205
#Precision@5:                  	0.0916
#True positives:               	16596
#False positives:              	272841
#False negatives:              	13042
#Documents evaluated:          	29643

# pecos / transformer
poetry run annif eval P279-xtransformer-en $inputdir
# zero scores on wiki5m

#poetry run annif eval P279-pav-en $inputdir
#sources=P279-mllm-en,P279-stwfsa-en
#Precision (doc avg):          	0.0894
#Recall (doc avg):             	0.8805
#F1 score (doc avg):           	0.1620
#Precision (subj avg):         	0.1036
#Recall (subj avg):            	0.1320
#F1 score (subj avg):          	0.1099
#Precision (weighted subj avg):	0.6911
#Recall (weighted subj avg):   	0.8807
#F1 score (weighted subj avg): 	0.7332
#Precision (microavg):         	0.0886
#Recall (microavg):            	0.8807
#F1 score (microavg):          	0.1611
#F1@5:                         	0.2912
#NDCG:                         	0.8485
#NDCG@5:                       	0.8457
#NDCG@10:                      	0.8485
#Precision@1:                  	0.8099
#Precision@3:                  	0.2881
#Precision@5:                  	0.1749
#True positives:               	26102
#False positives:              	268338
#False negatives:              	3536
#Documents evaluated:          	29643

#poetry run annif eval P279-nn-en $inputdir

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#warning: Unknown subject URI <http://www.wikidata.org/entity/Q5469882>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q6015104>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q6546278>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q677322>
#warning: Unknown subject URI <http://www.wikidata.org/entity/Q6964881>
