from corpus import *


def main():
    corpus = Corpus("test_corpus", "ansi")
    print(corpus.trigrams().frequencies().get_freq())


main()
