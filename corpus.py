"""
This file contains two classes that create a Corpus- and Subtitle-object. The Corpus-object is to be called to operate
on the subtitle corpus. It automatically creates a Subtitle-object for every subtitle file in the corpus and handles
them through its methods.
"""
from glob import glob
import typing
from typing import Any

import spacy
from nltk import trigrams
from nltk.tokenize import *
import re
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from numpy import ndarray, dtype, generic
import collections


class Subtitle:
    """
    The Subtitle-class represents the subtitles of a single movie. It presents multiple methods for operating on a
    subtitle file and saving them for later processing to reduce process time once you've done it.
    """

    def __init__(self, file_name: str, dir_length: int, encoding: str) -> None:
        """
        Stores file name and directory length.
        Then it extracts the text from a subtitle file and performs a fast cleaning (e.g. \\n, \\s)
        :param file_name: name of the subtitle file
        :param dir_length: length of the directory that contains the subtitle file
        :param encoding: encoding of the subtitle file
        :return None
        """
        self.__file_name = file_name[dir_length:-4]
        self.__dir_length = dir_length
        self.__encoding = encoding
        self.__text = " ".join([
            word.strip().lower()
            for word in open(file_name, "r", encoding=encoding).read().split()
        ])
        self.__ner_text = None
        self.__trigram_text = None

    def get_name(self) -> str:
        """
        Returns the name of the subtitle.
        :return: name of the subtitle
        """
        return self.__file_name

    def get_plain_text(self) -> str:
        """
        Returns the plain text of the subtitle.
        :return: text of the subtitle
        """
        return self.__text

    def get_ner_text(self) -> str:
        """
        Returns the NER processed text of the subtitle.
        :return: NER processed text
        """
        if self.__ner_text is None:
            raise ValueError("No NER text found")
        else:
            return self.__ner_text

    def get_trigram_text(self) -> str:
        """
        Returns the trigram processed text of the subtitle.
        :return: trigram processed text
        """
        if self.__trigram_text is None:
            raise ValueError("No trigram text found")
        else:
            return self.__trigram_text

    def clean(self, save: bool, output_path: str, encoding: str) -> None:
        """
        Cleans SRT-file by removing SRT-syntactic metadata (e.g. time to display text) and added information in brackets

        TO-DO: clean bracket information

        :return: None
        """
        self.__text = re.sub(r"(\d+\s*)?\d\d:\d\d:\d\d[,\.]\d\d\d|-->", "", self.__text)
        self.__text = re.sub(r"[!\"#$%&\'()*+,./:;<=>?@\[\]^_`{|}~\\-]", "", self.__text)
        self.__text = re.sub(r"\s\s+", " ", self.__text)
        if save:
            self.save_to_file(self.__text, output_path, "txt", encoding)

    def ner(self, save: bool, output_path: str) -> None:
        """
        Performs NER with SpaCy-Model "en_core_web_trf" and replaces words with entity type
        Example: "Sauron betrayed them all" --> "PERSON betrayed them all"]
        :param save: Whether the file is saved as pickle
        :param output_path: Output folder for saving file
        :return: None
        """
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("merge_entities")
        doc = nlp(self.__text)
        self.__ner_text = " ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc])
        if save:
            self.save_to_file(self.__text, output_path, "pkl")

    def create_trigrams(self, save: bool, output_path: str) -> None:
        """
        Tokenizes subtitle text and creates 3-grams from tokens
        :return: list of 3-grams
        """
        self.__trigram_text = trigrams(word_tokenize(self.__text))
        if save:
            self.save_to_file(list(self.__trigram_text), output_path, "pkl")

    def save_to_file(self, data: typing.Any, output_path: str, file_format: str, encoding: str = "utf-8") -> None:
        """
        Creates or uses given directory to save data to pickle file
        :param data: Any data to be saved
        :param output_path: name of output directory
        :param file_format: file format of output file (e.g. txt, pkl)
        :param encoding: encoding of the output file for non-binary data
        :return: None
        """
        path = os.path.join(output_path, f"{self.__file_name}.{file_format}")
        try:
            os.makedirs(output_path)
        except FileExistsError:
            pass

        if file_format == "pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        elif file_format == "txt":
            with open(path, "w", encoding=encoding) as f:
                f.write(data)

    def load_from_file(self, input_path: str, mode: str, encoding: str = "utf-8") -> None:
        """
        Loads subtitle that's been processed by class methods from pickle file
        :param input_path: Directory name of stored data
        :param mode: type of saving ("ner", "tri")
        :param encoding: encoding of the input file for non-binary data
        :return:
        """
        if mode is None:
            mode = ["txt", "ner", "tri"]
        if mode == "txt":
            path = os.path.join(input_path, f"{self.__file_name}.txt")
            with open(path, "r", encoding=encoding) as f:
                self.__text = f.read()
        elif mode == "ner":
            path = os.path.join(input_path, f"{self.__file_name}.pkl")
            with open(path, "rb") as f:
                self.__ner_text = pickle.load(f)
        elif mode == "tri":
            path = os.path.join(input_path, f"{self.__file_name}.pkl")
            with open(path, "rb") as f:
                self.__trigram_text = pickle.load(f)
        else:
            raise ValueError("Wrong parameter for mode. Permitted only 'txt', 'ner' or 'tri'")


class Corpus:
    """
    The Corpus-class represents a whole corpus of subtitles by creating Subtitle-objects for every file in corpus when
    called. This class should be used to operate on a corpus instead of directly calling the Subtitle-class. It presents
    all necessary methods to operate on all Subtitle-class methods
    """

    def __init__(self, dir_name: str, encoding: str = "utf-8", file_format: str = "srt") -> None:
        self.__dir_length = len(dir_name) + 1
        self.__file_names = glob(os.path.join(dir_name, f"*.{file_format}"))
        self.__encoding = encoding
        self.__subtitles = self.__create_subtitles()
        self.__freq = None

    def __create_subtitles(self) -> typing.List[Subtitle]:
        """
        Creates Subtitle-objects for every subtitle in corpus
        :return: list of Subtitle-Objects
        """
        return [Subtitle(
            file_name=name,
            dir_length=self.__dir_length,
            encoding=self.__encoding)
            for name in self.__file_names
        ]

    def get_subtitles(self) -> typing.List[Subtitle]:
        return self.__subtitles

    def get_text(self) -> typing.List[str]:
        return [
            subtitle.get_plain_text()
            for subtitle in self.__subtitles
        ]

    def get_mfw(self):
        return self.__freq

    def clean_subtitles(self, save: bool = False, output_path: str = "cleaned_output",
                        output_encoding: str = "utf-8") -> typing.Self:
        """
        Cleans whole corpus of subtitle files and writes it to txt file
        :param save: Whether the file is to be saved
        :param output_path: folder to write cleaned files
        :param output_encoding: encoding of txt output file
        :return: None
        """
        [
            subtitle.clean(save=save,
                           output_path=output_path,
                           encoding=output_encoding)
            for subtitle in self.__subtitles
        ]
        return self

    def load_cleaned_subtitle(self, input_path: str = "cleaned_output") -> typing.Self:
        [
            sub.load_from_file(input_path=input_path, mode="txt")
            for sub in self.__subtitles
        ]
        return self

    def ner(self, save: bool = True, output_path: str = "ner_output") -> typing.Self:
        """
        Performs NER on whole corpus and replaces entities with generic names
        \nOptional: Save results to default ("ner_output/") or user defined directory
        :param save: Whether the file is saved as pickle
        :param output_path: path to output directory in case of saving
        :return: None
        """
        i = 1
        for sub in self.__subtitles:
            print(str(i) + "/" + str(len(self.__subtitles)))
            sub.ner(
                save=save,
                output_path=output_path)
            i += 1
        return self

    def load_ner(self, input_path: str = "ner_output") -> typing.Self:
        """
        Retrieves subtitles that have already been processed with self.ner() from pickle files.
        :param input_path: Directory name where pickles are stored. Default equals default path from saving
        :return: List with loaded ner data as elements
        """
        [
            sub.load_from_file(input_path=input_path, mode="ner")
            for sub in self.__subtitles
        ]
        return self

    def trigrams(self, save: bool = True, output_path: str = "trigram_output") -> typing.Self:
        """
        Transforms whole corpus into 3-grams.
        \nOptional: Save results to default ("trigram_output/) or user defined directory
        :param save: Whether Data should be saved. Default: True
        :param output_path: Output directory to save corpus as 3-gram
        :return: Subtitles in 3-gram form
        """
        [
            sub.create_trigrams(
                save=save,
                output_path=output_path)
            for sub in self.__subtitles
        ]
        return self

    def load_trigrams(self, input_path: str = "trigram_output") -> typing.Self:
        """
        Retrieves subtitles that have already been processed with self.trigrams() from pickle files.
        :param input_path: directory name where pickles are stored. Default equals default path from saving
        :return: self
        """
        [
            sub.load_from_file(input_path=input_path, mode="tri")
            for sub in self.__subtitles
        ]
        return self

    def mfw(self, min_freq: int = 150, save: bool = True, output_path: str = "frequency_output") -> typing.Self:
        """
        Counts frequencies of 3-grams
        :param min_freq: minimum frequency of n-grams
        :param save: whether Data should be saved. Default: True
        :param output_path: output directory to save frequencies list
        :return: self
        """
        counter = collections.Counter()
        [
            counter.update(
                sub.get_plain_text().split()
            )
            for sub in self.__subtitles
        ]

        self.__freq = pd.Series(
            dict(counter),
            index=dict(counter).keys(),
            name="frequency"
        ).sort_values(ascending=False).iloc[:min_freq]

        if save:
            self.save_to_file(self.__freq, output_path, "mfw_list_all", "csv")
        return self

    def load_mfw(self) -> typing.Self:
        self.__freq = self.load_from_file("frequency_output", "mfw_list_all", "csv")
        return self

    def normalise(self) -> typing.Self:
        """
        Normalises frequencies
        :return: self
        """
        return self

    def culling(self) -> typing.Self:
        return self

    def cluster(self, trigrams: list) -> typing.Self:
        return self

    def save_to_file(self, data: pd.Series, output_path: str, file_name: str, file_format: str,
                     encoding: str = "utf-8") -> typing.Self:
        path = os.path.join(output_path, f"{file_name}.{file_format}")
        try:
            os.makedirs(output_path)
        except FileExistsError:
            pass

        if file_format == "csv":
            data.to_csv(path)

        return self

    def load_from_file(self, input_path:str, file_name: str, file_format: str) -> typing.Self:
        path = os.path.join(input_path, f"{file_name}.{file_format}")
        if file_format == "csv":
            self.__freq = pd.read_csv(path)
        return self
