import argparse
from model import NGramModel
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_path', required=True,
                        help='Path to the directory with model')
    parser.add_argument('--prefix', dest='prefix', required=False, nargs='+',
                        help='Starting words for the model')
    parser.add_argument('--length', dest='length', required=True, type=int,
                        help='Final length of expected text')
    args = parser.parse_args()
    model = NGramModel(n=3, model_path=args.model_path)
    prefix = None
    if args.prefix is not None:
        prefix = " ".join(args.prefix)
    t1 = time.time()
    model.generate(length=args.length, prefix=prefix)
    t2 = time.time()
    print("Time of generation:\n{:2.4f} sec".format(t2 - t1))
