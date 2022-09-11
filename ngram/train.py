import argparse
from model import NGramModel
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', dest='data_path', required=False,
                        help='Path to the directory with collection of text files')
    parser.add_argument('--model', dest='model_path', required=True,
                        help='Path to the directory for saving the model')
    args = parser.parse_args()
    model = NGramModel(n=3, model_path=args.model_path)
    t1 = time.time()
    model.fit(data_path=args.data_path)
    t2 = time.time()
    print("Time of training: {:2.4f} sec".format(t2 - t1))
