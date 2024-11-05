#  Natural Language Suggestion Generation for Software Vulnerability Repair

## Requirements
+ importlib-metadata<5.0  
+ spacy==3.0.6  
+ bert-score==0.3.11  
+ datasets==2.5.1  
+ gym==0.21.0
+ jsonlines==3.0.0
+ nltk==3.7
+ pandas==1.3.5
+ rich==12.0.0
+ stable-baselines3==1.5.1a5
+ torch==1.11.0
+ torchvision==0.12.0
+ tqdm==4.64.0
+ transformers==4.18.0
+ wandb==0.12.15
+ jsonlines==3.0.0
+ rouge_score==0.0.4
+ sacrebleu==2.2.0
+ py-rouge==1.1

## Dataset
We provide the constructed dataset in the directory `data`, which includes 18,517 pairs of vulnerable functions and suggestions, along with the patches. We randomly split them into `train`, `valid`, and `test` datasets with the fraction of 8:1:1. As the filename indicates, for example, files `train.csv` and `valid.csv` contain samples for training and validation respectively. In each file, there are three columns "id", "function", "suggestion", and "patch", which are the vulnerable functions, the corresponding suggestions, and the patches respectively.

## How to Run
1. `cd data` and `unzip file.zip -d ./` to extract the data files;
2. `python base_tira.py` for preprocessing, training and fine-tuning;
3. `python tira_infer.py` for inference of our model;
4. `python evaluation.py` for getting the results of all metrics.
