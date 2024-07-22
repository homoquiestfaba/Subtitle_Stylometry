import spacy
import typing
from glob import glob
import pickle


def load_data(name: str) -> typing.AnyStr:
    return " ".join([
        word.strip()
        for word in open(name, "r", encoding="utf-8").read().split()
    ])


def ner(text: str) -> typing.List[str]:
    """
    Performs NER with SpaCy-Model "en_core_web_trf" and replaces words with entity type
    Example: "Sauron betrayed them all" --> ["PERSON", "betrayed", "them", "all"]
    :param text: subtitle text of one movie
    :return: subtitle with replaces entity types
    """
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("merge_entities")
    doc = nlp(text)
    return [t.text if not t.ent_type_ else t.ent_type_ for t in doc]


def write_pickle(ents: list, name: str) -> typing.NoReturn:
    with open("test_pickle/" + name + ".pkl", "wb") as f:
        pickle.dump(ents, f)


def main():
    file_names = glob("test_corpus/*.txt")
    for name in file_names:
        text = load_data(name)
        print(text)
        ner_text = ner(text)
        print(ner_text)
        write_pickle(ner_text, name[12:-4])
        quit()


main()
