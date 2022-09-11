import os
import argparse
import torch
from model import TextGenModel, init_state
from train import normalize_text
import torch.nn.functional as F
import json
import numpy as np
import random
from typing import List


def upload_data(model_path: str):
    """
    Upload data that will be used during text generation by the model.

    :param str model_path: Path to the model location

    """
    if not os.path.exists("useful_data.json"):
        print("Could not find file with data for model usage")
        exit(1)
    with open("useful_data.json") as f:
        useful_data = json.load(f)
    vocab = useful_data['vocab']
    model = TextGenModel(vocab_size=len(vocab),
                         embedding_dim=useful_data["embedding_dim"],
                         lstm_size=useful_data["lstm_size"],
                         lstm_layers_num=useful_data["lstm_layers_num"],
                         dropout_rate=useful_data["dropout_rate"]
                         )
    model.load_state_dict(torch.load(model_path))
    return model, vocab, useful_data


def generate(model_path: str,
             prefix: List[str],
             length: int):
    """
    Generate text based on prefix.

    :param str model_path: Path where the model will be saved after training
    :param List[str] prefix: List of words that will be used as the beginning of future text
    :param int length: Length of future text

    """
    model, vocab, useful_data = upload_data(model_path)
    inv_vocab = {v: k for k, v in vocab.items()}
    if prefix is None or len(prefix) == 0:
        prefix = [random.choice(list(vocab.keys()))]
    prefix_text = " ".join(prefix).lower()
    normalized_prefix = normalize_text(prefix_text)
    if len(normalized_prefix) == 0:
        print("No text for the model")
        exit(1)

    model.eval()
    result_string = normalized_prefix + " "
    context = []
    prefix_words = normalized_prefix.split()
    prefix_length = len(prefix_words)
    for word in prefix_words:
        if word in vocab.keys():
            context.append(vocab[word])
        else:
            context.append(vocab['UNK'])
    h, c = init_state(useful_data['lstm_layers_num'], 1, useful_data['lstm_size'])
    while len(context) < length:
        context_tensor = torch.tensor([context])
        predictions, (h, c) = model(context_tensor, (h, c))
        weights = F.softmax(predictions[0][-1], dim=0).detach().numpy()
        word_index = np.random.choice(len(vocab), p=weights)
        context.append(word_index)
    result_string += " ".join([inv_vocab[word_idx] for word_idx in context[prefix_length:]])
    print(f"Used prefix:\n{normalized_prefix}")
    print(f"Generated text:\n{result_string}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_path', required=True,
                        help='path to the directory with model')
    parser.add_argument('--prefix', dest='prefix', required=False, nargs='+',
                        help='starting words for the model')
    parser.add_argument('--length', dest='length', required=True, type=int,
                        help='final length of expected text')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        print("Could not find model")
        exit()
    if args.length <= 0:
        print("Sentence length must be positive")
        exit()
    generate(model_path=args.model_path, prefix=args.prefix, length=args.length)
