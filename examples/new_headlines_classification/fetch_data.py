import argparse
import json
import os
import re
import random

import requests
import sentencepiece as spm


def sentence_preprocess(s: str):
    s = "".join(i for i in s.lower() if i.isalpha() or i == " ")
    s = re.sub(r"\s+", " ", s)
    return s


def download_raw_data():
    RAW_DATA_URL = "https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/raw/master/Sarcasm_Headlines_Dataset.json"
    response = requests.get(RAW_DATA_URL)

    assert response.status_code == 200
    data = response.content.decode("utf-8")

    return [json.loads(line.strip()) for line in data.split("\n") if len(line) > 1]


def build_spm_model(model_prefix: str, vocab_size: int, data: list[dict]):
    src = list(map(lambda x: x["headline"], data))

    spm.SentencePieceTrainer.train(
        model_prefix=model_prefix,
        sentence_iterator=map(sentence_preprocess, src),
        vocab_size=vocab_size,
        model_type="bpe",
    )


def tokenize(model, sents: list[str]):
    return [model.encode(sent, out_type=str) for sent in sents]


def process_data(raw_data: list[dict], model_path: str):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    tokenized_data = tokenize(sp, [sample["headline"] for sample in raw_data])

    return [(" ".join(sent), doc["is_sarcastic"]) for sent, doc in zip(tokenized_data, raw_data)]


def write_dataset(prefix: str, dataset: list[tuple[str, int]]):
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    with (
        open(f"{prefix}/src.txt", "wt", encoding="utf-8") as src_f,
        open(f"{prefix}/tgt.txt", "wt", encoding="utf-8") as tgt_f
    ):
        for src_line, label in dataset:
            src_f.write(src_line + "\n")
            tgt_f.write(str(label) + "\n")

def positive_ratio(dataset):
    return round(sum(map(lambda x: x[1], dataset)) / len(dataset), 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-prefix", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    args = parser.parse_args()

    raw_data = download_raw_data()
    build_spm_model(args.model_prefix, args.vocab_size, raw_data)

    clean_data = process_data(raw_data, f"{args.model_prefix}.model")
    random.shuffle(clean_data)

    idx = int(len(clean_data) * args.train_ratio)
    print("Train cut-off index:", idx)
    train_set = clean_data[None:idx]
    valid_set = clean_data[idx:None]

    print("Positive ratio for train:", positive_ratio(train_set))
    print("Positive ratio for valid:", positive_ratio(valid_set))

    write_dataset(os.path.join(args.output_folder, "train"), train_set)
    write_dataset(os.path.join(args.output_folder, "valid"), valid_set)
