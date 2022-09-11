import sys
import os
import re
from collections import deque, defaultdict
import numpy as np
import pickle
from typing import List


def init_dict():
    return dict()


class NGramModel:
    def __init__(self,
                 n: int,
                 model_path: str):
        """
        Create instance of NGramModel - text generator based on n-grams.

        :param int n: Size of each n-gram
        :param str model_path: Path to the model location

        """
        self.n = n
        self.model_path = model_path
        self.model = defaultdict(init_dict)

    def __read_text(self, data_path: str):
        """
        Read all text files in order to prepare data for the model training.

        :param data_path: Path to the directory where all text files are located

        """
        if data_path is not None and not os.path.exists(data_path):
            print("Error: could not find directory with text data for training")
            exit(1)
        text = ""
        if data_path is None:
            text = sys.stdin.read()
        else:
            filenames = next(os.walk(data_path), (None, None, []))[2]
            for filename in filenames:
                filepath = os.path.join(data_path, filename)
                with open(filepath, 'rt') as f:
                    text += f.read()
                text += " "
        return text

    def __get_tokens(self, text: str):
        """
        Return tokens of the input text - Russian words, numbers and punctuation marks.

        :param str text: Input text that is going to be divided into tokens

        """
        # Divide into tokens leaving only Russian words, numbers amd punctuation marks
        reg = re.compile(r'[ёа-яА-Я0-9-]+|[.,:;?!]+')
        tokens = reg.findall(text)

        # Remove repetitive punctuation marks that can occur after tokenization
        reg = re.compile(r'[.,:;?!]+')
        filtered_tokens = []
        previous_is_punctuation = False
        for token in tokens:
            obj = reg.match(token)
            if obj is None:
                filtered_tokens.append(token)
                previous_is_punctuation = False
            else:
                if not previous_is_punctuation:
                    previous_is_punctuation = True
                    filtered_tokens.append(token)
        return filtered_tokens

    def __get_ngrams(self, tokens: List[str]):
        """
        Generate n-grams of the input token array.

        :param List[str] tokens: Input list of tokens

        """
        ngram = deque(tokens[:self.n])
        yield ngram
        for i in range(self.n, len(tokens)):
            ngram.popleft()
            ngram.append(tokens[i])
            yield ngram

    def load_model(self):
        """
        Load model from the path that was previously defined in the class constructor.

        """
        if not os.path.exists(self.model_path):
            print("Error: could not find the model because of incorrect path")
            exit(1)
        else:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

    def dump_model(self):
        """
        Dump/save model to the path that was previously defined in the class constructor.

        """
        directory, _ = os.path.split(self.model_path)
        if len(directory) > 0 and not os.path.exists(directory):
            print("Error: could not find directory for saving the model")
            exit(1)
        else:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

    def fit(self, data_path: str):
        """
        Train the model on data from all text files located in data_path.

        :param str data_path: Path to the collection of text files

        """
        # Read text files
        text = self.__read_text(data_path)
        tokens = self.__get_tokens(text)
        if len(tokens) < self.n:
            print("Error: provided text data is not enough for training with current value of parameter n")
            exit(1)

        # Count number of n-gram and (n-1)-gram occurrences in the tokenized text
        # in order to calculate next token probability in the future
        nlessgrams_count = defaultdict(lambda: 0)
        ngrams_count = defaultdict(lambda: 0)
        for ngram in self.__get_ngrams(tokens):
            ngram_tuple = tuple(ngram)
            nlessgrams_count[ngram_tuple[:-1]] += 1
            ngrams_count[ngram_tuple] += 1

        # Calculate probability of each possible next token based on the current (n-1)-gram
        for ngram, ngram_count in ngrams_count.items():
            ngram_tuple = tuple(ngram)
            nlessgram, last_token = ngram_tuple[:-1], ngram_tuple[-1]
            self.model[nlessgram][last_token] = ngram_count / nlessgrams_count[nlessgram]

        # Save the model
        self.dump_model()
        print("Model has been successfully trained and saved to the defined directory")

    def generate(self,
                 length: int,
                 prefix: str):
        """
        Generate text of the defined length. Can be executed with the prefix defined by the user.

        :param int length: Final length of expected text (unit of counting - word or punctuation mark)
        :param str prefix: The beginning of future text defined by the user. Can be empty.

        """
        # Load model if we do not have it at this moment
        if length <= 0:
            print("Error: value of text length must be positive")
            exit(1)
        if len(self.model.keys()) == 0:
            self.load_model()

        # Process input prefix. If user did not define prefix, we randomly choose one capitalized word
        # that will be the prefix
        reg = re.compile(r'[.?!]+')
        tokens_seq = []
        if prefix is None:
            nlessgrams = list(filter(lambda x: reg.match(x[-2]) is not None, self.model.keys()))
            nlessgram = nlessgrams[np.random.randint(0, len(nlessgrams))]
            tokens_seq.append(nlessgram[-1].capitalize())
            prefix = tokens_seq[0]
        else:
            tokens = self.__get_tokens(prefix)
            tokens[0] = tokens[0].capitalize()
            tokens_seq.extend(tokens)

        # Prepare first (n-1)-gram. If prefix has more tokens than n-1,
        # we locate in the first gram only n-1 last tokens. If prefix length is shorter than n-1, we locate
        # all prefix in the end of (n-1)-gram - the beginning will be filled in code below.
        index = len(tokens_seq) - self.n + 1
        if index < 0:
            index = 0
        nlessgram = deque(tokens_seq[index:])
        first_token_idx = 1
        if len(nlessgram) != self.n - 1:
            first_token_idx = self.n - len(tokens_seq) - 1
            nlessgram.extendleft(['#'] * first_token_idx)

        # New tokens generation.
        # If we run into unknown (n-1)-gram (if it is not present in the model - in most cases it
        # happens only in the begging during prefix processing), we shorten (n-1)-gram (prefix) from the
        # left side and try to find it in the model again.
        # When we find current (n-1)-gram in the model, we get possible next tokens and choose randomly one of them
        # due to calculated probability distribution.
        while len(tokens_seq) < length:
            key_tuple = tuple(nlessgram)
            if key_tuple not in self.model.keys():
                while first_token_idx != self.n - 1:
                    other_keys = list(filter(lambda x: x[first_token_idx:] == key_tuple[first_token_idx:],
                                             self.model.keys()))
                    if len(other_keys) > 0:
                        key_tuple = other_keys[np.random.randint(0, len(other_keys))]
                        nlessgram = deque(key_tuple)
                        break
                    first_token_idx += 1
                else:
                    other_keys = list(filter(lambda x: reg.match(x[-1]) is None, self.model.keys()))
                    key_tuple = other_keys[np.random.randint(0, len(other_keys))]
                    nlessgram = deque(key_tuple)
            possible_tokens = np.array(list(self.model[key_tuple].keys()))
            probabilities = np.array(list(self.model[key_tuple].values()))
            next_token = np.random.choice(possible_tokens, p=probabilities)
            nlessgram.popleft()
            nlessgram.append(next_token)
            tokens_seq.append(next_token)

        # Create resulted text and print it
        result_string = " ".join(tokens_seq)
        result_string = re.sub(r"(\s)([!?.:;,]+)", r"\2", result_string)
        print(f"Used prefix:\n{prefix}")
        print(f"Generated string:\n{result_string}")
