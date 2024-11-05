from tqdm import tqdm
from evaluate import load as load_metric
import torch
from joblib import Parallel, delayed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reweight import *
from nltk.tokenize import word_tokenize


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return model, tokenizer


def generate_predictions(texts, batch_size=64):
    predictions = []

    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        tokenized_functions = Parallel(n_jobs=16)(
            delayed(tokenize_code)(function) for function in batch_texts
        )

        model_inputs = tokenizer(tokenized_functions, is_split_into_words=True, return_tensors="pt", padding="longest",
                                 truncation=True).to(device)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].float()


        batch_outputs = model.generate(**model_inputs, max_new_tokens=100, num_beams=5)

        batch_preds = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)

    return predictions

def compute_metrics(predictions, references):
    bleu = load_metric("sacrebleu")
    rouge = load_metric("rouge")
    # predictions = [p.lower() for p in predictions]
    # references = [r.lower() for r in references]
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])["score"]
    rouge_score = rouge.compute(predictions=predictions, references=references)

    metrics = {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_score["rouge1"],
        "ROUGE-2": rouge_score["rouge2"],
        "ROUGE-L": rouge_score["rougeL"]
    }
    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    model_path = "results_tira/checkpoint-13890"
    test_csv_path = "../data/test.csv"

    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    test_df = pd.read_csv(test_csv_path)
    vectorizer = train_or_load()
    # Assuming the columns are named 'function' and 'suggestion'
    input_texts = test_df['function'].tolist()
    reference_texts = test_df['suggestion'].tolist()

    predictions = generate_predictions(input_texts)
    metrics = compute_metrics(predictions, reference_texts)

    # Optionally, save the predictions to a file
    output_file_path = "base_tira.out"
    with open(output_file_path, 'w') as f:
        for prediction in predictions:
            f.write("%s\n" % prediction)

    print(metrics)
