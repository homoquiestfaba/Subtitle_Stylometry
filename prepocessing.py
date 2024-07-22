from nltk import trigrams
from nltk.tokenize import *
from glob import glob
import typing


def load_data(name: str) -> typing.AnyStr:
    return "".join([
        char
        for char in open(name, "r", encoding="utf-8").read()
        if char != "\n"
    ])


def trigram(text: str) -> typing.List[tuple]:
    trigram_list = list(trigrams(word_tokenize(text)))
    return trigram_list


def main():
    file_names = glob("test_corpus/*.txt")
    for name in file_names:
        text = load_data(name)
        trigram_list = list(trigrams(word_tokenize(text)))


main()
