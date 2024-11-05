from datasets import load_dataset
from evaluate import load as load_metric
from joblib import Parallel, delayed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,  Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

from modeling_codet5 import CodeT5ForConditionalGeneration
from reweight import *
import spacy
import torch

nlp = spacy.load("en_core_web_sm")

def create_ra_masks(texts, tokenizer):
    # Tokenize all texts at once
    docs = [nlp(text) for text in texts]
    word_lists = [[token.text for token in doc] for doc in docs]
    encoding = tokenizer(text_target=word_lists, is_split_into_words=True, padding=True, truncation=True, max_length=100)

    ra_masks = []

    for i, doc in enumerate(docs):
        # Initialize all ra masks to 0 (ignore all tokens initially)
        ra_mask = [0]*len(encoding['attention_mask'][i])

        word_ids = encoding.word_ids(batch_index=i)  # Get word_ids for the specific batch index

        # Identify verbs and their objects using spaCy's parse
        for token in doc:
            if token.pos_ == 'VERB':
                verb_idx = token.i
                for child in token.children:
                    if child.dep_ in ['dobj', 'iobj', 'pobj', 'nsubjpass']:
                        obj_idx = child.i
                        # Find tokens corresponding to verb and object
                        verb_tokens = [idx for idx, word_id in enumerate(word_ids) if word_id == verb_idx]
                        obj_tokens = [idx for idx, word_id in enumerate(word_ids) if word_id == obj_idx]

                        # Set ra mask to 1 for verb-object tokens
                        for vt in verb_tokens:
                            ra_mask[vt] = 1  # Attend to these tokens
                        for ot in obj_tokens:
                            ra_mask[ot] = 1  # Attend to these tokens

        ra_masks.append(ra_mask)

    return encoding["input_ids"], ra_masks


# Tokenization function
def tokenize_function_and_suggestion(examples):
    tokenized_functions = Parallel(n_jobs=16)(
        delayed(tokenize_code)(function) for function in examples["function"]
    )
    tfidf_matrix = vectorizer.transform([' '.join(func) for func in tokenized_functions])

    model_inputs = tokenizer(tokenized_functions, is_split_into_words=True, return_tensors="pt", padding="longest", truncation=True)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].float()
    for idx in range(len(examples)):
        word_ids = model_inputs.word_ids(batch_index=idx)
        antlr_tokens = tokenized_functions[idx]
        # print(model_inputs["attention_mask"][idx])
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                antlr_token = antlr_tokens[word_id]
                token_weight = tfidf_matrix[idx, vectorizer.vocabulary_.get(antlr_token, 0)]
                model_inputs["attention_mask"][idx][i] += token_weight
        # print(model_inputs["attention_mask"][idx])

    labels, ra_masks = create_ra_masks(examples["suggestion"], tokenizer)
    model_inputs["labels"] = labels
    model_inputs["ra_mask"] = ra_masks
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__ == "__main__":
    # Load the datasets
    data_files = {
        "train": "../data/train.csv",
        "valid": "../data/valid.csv",
    }
    raw_datasets = load_dataset('csv', data_files=data_files)

    # Initialize the tokenizer
    model_checkpoint = "Salesforce/codet5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    vectorizer = train_or_load()

    # Tokenize all datasets
    tokenized_datasets = raw_datasets.map(tokenize_function_and_suggestion, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'ra_mask'])

    # Load the model
    model = CodeT5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_tira",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        num_train_epochs=30,
        weight_decay=0.01,
        save_total_limit=10,
        generation_max_length=100,
        remove_unused_columns=False
    )

    # Define the compute_metrics function for evaluation
    metric = load_metric("sacrebleu")

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

