from flair.data import Sentence
from flair.models import SequenceTagger
import re
from os import listdir
import pickle

FILE_NAMES = []


def get_names(path: str):
    """
    :param path:
    :return:
    """
    global FILE_NAMES
    FILE_NAMES = [f for f in listdir(path)]
    return FILE_NAMES


def retrieve_text(file_name: str):
    with open("test_corpus/" + file_name, "r", encoding="utf-8") as f:
        text_raw = f.readlines()
    text = ""
    for line in text_raw:
        if line[:-1]:
            text += line[:-1] + " "
    return text[:-1]


def write_pickle(ents: list):
    # FILE_NAMES[0][:-4]
    with open("test" + ".pkl", "wb") as f:
        pickle.dump(ents, f)


def ner_with_flair(text: str):
    """
    Performs Named Entity Recognition on a subtitle file
    :param text: subtitle in string form
    :return: Not sure yet
    """
    # Load the German language model
    # optional: flair/ner-german-large (F1 score: 0.92), flair/ner-german-legal, flair/ner-german
    # test = "I am Optimus Prime and I send this message to any surviving autobots taking refuge among the stars. We are here. We are waiting."
    tagger = SequenceTagger.load("flair/ner-german-large")
    sentence = Sentence(text)
    tagger.predict(sentence)
    # print(sentence)

    ents = []
    for entity in sentence.get_spans('ner'):
        token = str(entity.tokens)
        token = token.split(",")
        token_ind = []
        for tok in token:
            tok = re.sub(r".*\[|\].*", "", tok)
            token_ind.append(int(tok))
        # print(type(entity.tokens))
        ent = str(entity.get_labels()[0]).split()[-2]
        print(token_ind)
        print(ent)
        ents.append([token_ind, ent])
        """if re.search(r"LOC", ent):
            print(ent)
            loc = re.search(r'".+"', ent).group(0)
            loc = re.sub(r'"', "", loc)
            locations.append(loc)"""
    write_pickle(ents)


def listify_test(text: str):
    text = re.split(r"([\.,\!\s:])", text)
    text_list = []
    for word in text:
        if word and word != " ":
            text_list.append(word)
    print(text_list)
    return text_list


def load_pickle():
    with open("test.pkl", "rb") as f:
        ents = pickle.load(f)
        print(ents)
    return ents


def tag_text(text: list, ner_ind: list):
    for ent in ner_ind:
        print(ent)
        for i in ent[0]:
            print(i)
            text[i] = ent[1]
    print(text)


def main():
    files = get_names("test_corpus")
    print(files)
    text = retrieve_text(files[0])
    print(text)
    # text = "Heyho"
    # ner_with_flair(text)
    ents = load_pickle()
    text = listify_test(text)
    tag_text(text, ents)


main()
