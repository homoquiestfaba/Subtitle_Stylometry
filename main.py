import os

from corpus import *


def main():
    corpus = Corpus("test_corpus_clean", file_format="txt")
    a = corpus.burrows_delta().get_delta()
    print(a)


if __name__ == '__main__':
    main()
