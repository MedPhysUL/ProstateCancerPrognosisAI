import json

from src.evaluation import PredictionComparator

bayes_seq_net_pred = json.load(open(r"local_data\preds\holdout\BayesSeqNet.json", "r"))
print(bayes_seq_net_pred)
