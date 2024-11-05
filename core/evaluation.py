import pandas as pd
from bert_score import BERTScorer
from evaluate import load
from bar import behavioral_acc
import sys
import json

df = pd.read_csv("../data/test.csv")

refs = list(df["suggestion"])


with open("base_tira.out") as f:
    lines = [line for line in f.readlines() if line.strip()]

scores = [behavioral_acc(r, p) for r, p in zip(refs, lines)]
print(sum(scores)/len(scores))
bleu = load("sacrebleu")
rouge = load("rouge")

bleu_score = bleu.compute(predictions=lines, references=[[ref] for ref in refs])["score"]
rouge_score = rouge.compute(predictions=lines, references=refs)

print(bleu_score, rouge_score)
scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score(lines, refs)
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
