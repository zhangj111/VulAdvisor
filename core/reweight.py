import os
import math
from joblib import dump, load
from antlr4 import *
from CPP14Lexer import CPP14Lexer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tokenize_code(code):
    lexer = CPP14Lexer(InputStream(code))
    stream = CommonTokenStream(lexer)
    stream.fill()
    tokens = [token.text for token in stream.tokens[:-1] if not token.text.isspace() and token.type != -1]
    return tokens

def train_or_load():
    model_path = "results_token/tfidf.joblib"
    if os.path.exists(model_path):
        return load(model_path)
    data_files = {
        "train": "../../data/train.csv",
        "valid": "../../data/valid.csv",
        "test": "../../data/test.csv"
    }

    corpus = []
    for file_type, file_path in data_files.items():
        df = pd.read_csv(file_path)
        corpus.extend([' '.join(tokenize_code(f)) for f in df['function'].tolist()])

    print(f"Corpus size: {len(corpus)}")
    vectorizer = TfidfVectorizer(lowercase=False, tokenizer=str.split)
    vectorizer.fit(corpus)

    dump(vectorizer, model_path)
    return vectorizer

def normalize_scores(scores):
    total = sum(scores)
    return [score / total for score in scores if total > 0]

def calculate_entropy(scores):
    return -sum(p * math.log2(p) for p in scores if p > 0)


def select_top_k_percent_lines(line_scores, k_percent=80):
    k = max(int(len(line_scores) * (k_percent / 100)), 5)
    top_k_scores = sorted([score for _, score in line_scores], reverse=True)[:k]
    k_score_threshold = top_k_scores[-1]
    top_k_lines = [line for line, score in line_scores if score >= k_score_threshold]

    return top_k_lines


def get_topk_lines(index_to_token, tfidf_matrix, tokenized_funcs):
    line_funcs = []

    # Iterate over each row (document) in the tfidf_matrix
    for i in range(tfidf_matrix.shape[0]):
        # Extract the row as a sparse matrix
        row = tfidf_matrix.getrow(i)

        # Convert the row to a (col_index, score) format
        col_indices = row.indices
        scores = row.data

        # Map column indices to tokens and create a {token: score} dict
        token_scores = {index_to_token[idx]: score for idx, score in zip(col_indices, scores)}

        line_scores = []
        current_line = []
        current_score = []
        token_list = tokenized_funcs[i].split()
        for token in token_list:
            current_line.append(token)
            current_score.append(token_scores.get(token, 0))
            if token in [';', '{', '}']:
                line_scores.append((' '.join(current_line), sum(current_score)/len(current_score)))
                current_line = []
                current_score = []

        if current_line:  # Handle any remaining tokens
            line_scores.append((' '.join(current_line), sum(current_score)/len(current_score)))

        line_funcs.append('\n'.join(select_top_k_percent_lines(line_scores)))

    return line_funcs



def select_top_k_percent_lines_entropy(lines, line_entropies, k_percent):
    k = int(len(lines) * (k_percent / 100))
    indexed_entropies = list(enumerate(line_entropies))
    indexed_entropies.sort(key=lambda x: x[1], reverse=True)
    top_k_lines = [lines[index] for index, _ in indexed_entropies[:k]]
    return top_k_lines


if __name__ == '__main__':
    train_or_load()