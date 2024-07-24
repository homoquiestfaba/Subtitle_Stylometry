import os

from corpus import *


def main():
    corpus = Corpus("test_corpus_clean", file_format="txt")
    a = corpus.ner()
    print(a)


if __name__ == '__main__':
    main()
