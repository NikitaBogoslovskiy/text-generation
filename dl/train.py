import torch
from model import TextGenModel, init_weights, init_state
import torch.nn.functional as F
import os
import re
import argparse
import sys
import json


class NGramBlock:
    def __init__(self, context, target):
        self.context = context
        self.target = target


def read_text(path=None):
    text = ""
    if path is None:
        text = sys.stdin.read().lower()
    else:
        filenames = next(os.walk(path), (None, None, []))[2]
        for filename in filenames:
            filepath = os.path.join(path, filename)
            with open(filepath, 'rt') as f:
                text += f.read().lower()
            text += " "
    return text


def normalize_text(text):
    new_text = ""
    for i in range(len(text)):
        ch = text[i]
        if re.match(r'[а-яА-Я\s]', str(ch)):
            new_text += str(ch)
    new_text = re.sub('\n', ' ', new_text)
    new_text = re.sub('\xa0', ' ', new_text)
    new_text = re.sub(' +', ' ', new_text).strip()
    return new_text


def get_vocab_and_ngram(text, window_size):
    words = text.split()
    words.append("UNK")
    vocab = {word: i for i, word in enumerate(set(words))}
    ngram = []
    for i in range(window_size, len(words), window_size):
        extended_window = list(map(lambda x: vocab[x], words[i-window_size:i+1]))
        block = NGramBlock(
            context=extended_window[:-1],
            target=extended_window[1:]
        )
        ngram.append(block)
    return vocab, ngram


def prepare_data(ngram, batch_size, train_coef=0.99):
    train_contexts = []
    train_targets = []
    overall_amount = len(ngram)
    train_amount = int(overall_amount * train_coef // batch_size) * batch_size

    for i in range(0, train_amount, batch_size):
        contexts = list(map(lambda x: x.context, ngram[i:i+batch_size]))
        targets = list(map(lambda x: x.target, ngram[i:i+batch_size]))
        context_tensor = torch.tensor(contexts)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        train_contexts.append(context_tensor)
        train_targets.append(target_tensor)

    context = list(map(lambda x: x.context, ngram[train_amount:]))
    target = list(map(lambda x: x.target, ngram[train_amount:]))
    test_context = torch.tensor(context)
    test_target = torch.tensor(target, dtype=torch.long)

    return train_contexts, train_targets, test_context, test_target


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model has been saved to "{path}"')


def save_useful_data(useful_data):
    path = "useful_data.json"
    with open(path, 'wt') as f:
        json.dump(useful_data, f)


def train(model_path, data_path=None, window_size=6, batch_size=500):
    text = read_text(path=data_path)
    normalized_text = normalize_text(text=text)
    if len(normalized_text) == 0:
        print("No text for the model")
        exit(1)
    vocab, ngram = get_vocab_and_ngram(normalized_text, window_size)
    train_contexts, train_targets, test_context, test_target = prepare_data(ngram, batch_size)

    embedding_dim = 128
    lstm_size = 128
    lstm_layers_num = 1
    dropout_rate = 0
    lr = 1e-3
    model = TextGenModel(vocab_size=len(vocab),
                         embedding_dim=embedding_dim,
                         lstm_size=lstm_size,
                         lstm_layers_num=lstm_layers_num,
                         dropout_rate=dropout_rate
                         )
    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print("=" * 20)
    print("Training process")
    print("=" * 20)
    model.train()
    num_batches = len(train_contexts)
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        print("-" * 20)
        print(f"Epoch {epoch}")
        print("-" * 20)
        h, c = init_state(lstm_layers_num, batch_size, lstm_size)
        for batch in range(num_batches):
            context = train_contexts[batch]
            target = train_targets[batch]
            predictions, (h, c) = model(context, (h, c))
            h = h.detach()
            c = c.detach()
            loss = F.cross_entropy(predictions.transpose(1, 2), target)
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(f"epoch {epoch}, batch {batch+1}: loss = {loss.item()}")
    print("=" * 20)
    print("End of training process")
    print("=" * 20)

    print("=" * 20)
    print("Testing... ", end='')
    h, c = init_state(lstm_layers_num, len(test_context), lstm_size)
    test_predictions, _ = model(test_context, (h, c))
    loss = F.cross_entropy(test_predictions.transpose(1, 2), test_target)
    print("Successfully finished")
    print(f"Eventual loss = {loss.item()}")
    print("=" * 20)
    save_model(model, model_path)
    data = {
        "vocab": vocab,
        "embedding_dim": embedding_dim,
        "lstm_size": lstm_size,
        "lstm_layers_num": lstm_layers_num,
        "dropout_rate": dropout_rate
    }
    save_useful_data(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', dest='data_path', required=False,
                        help='path to the directory with collection of text files')
    parser.add_argument('--model', dest='model_path', required=True,
                        help='path to the directory for saving the model')
    args = parser.parse_args()
    if args.data_path is not None and not os.path.exists(args.data_path):
        print("Could not find directory with text files")
        exit(1)
    directory, _ = os.path.split(args.model_path)
    if len(directory) > 0 and not os.path.exists(directory):
        print("Could not find directory for saving model")
        exit(1)
    train(data_path=args.data_path, model_path=args.model_path)
