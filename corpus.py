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
from scipy.stats import zscore


class Subtitle:
    """
    The Subtitle-class represents the subtitles of a single movie. It presents multiple methods for operating on a
    subtitle file and saving them for later processing to reduce process time once you've done it.
    """

    def __init__(self, file_name: str, dir_length: int, encoding: list) -> None:
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
        valid_encoding = False
        for enc in encoding:
            try:
                self.__text = " ".join([
                    word.strip().lower()
                    for word in open(file_name, "r", encoding=enc).read().split()
                ]
                )
                valid_encoding = True
                break
            except UnicodeDecodeError:
                pass
        if not valid_encoding:
            raise EncodingWarning(f"File {file_name} has invalid encoding")
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
        self.__text = re.sub(r"<.*>", "", self.__text)
        self.__text = re.sub(r"\(.*\)", "", self.__text)
        if save:
            self.__save_to_file(self.__text, output_path, "txt", encoding)

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
            self.__save_to_file(self.__text, output_path, "pkl")

    def count_tokens(self, mfw: pd.Series) -> pd.Series:
        """
        Counts all tokens of the subtitle file that are in the most frequent word list
        :param mfw: most frequent word list from whole corpus
        :return: pd.Series of token frequencies
        """
        if self.__ner_text:
            text = self.__ner_text
        else:
            text = self.__text
        total_freq = dict(collections.Counter(text.split()))
        freqs = {}
        for term in mfw.index:
            try:
                freqs[term] = total_freq[term]
            except KeyError:
                freqs[term] = 0
        return pd.Series(freqs.values(), index=mfw.index)

    def create_trigrams(self, save: bool, output_path: str) -> None:
        """
        Tokenizes subtitle text and creates 3-grams from tokens
        :return: list of 3-grams
        """
        self.__trigram_text = trigrams(word_tokenize(self.__text))
        if save:
            self.__save_to_file(list(self.__trigram_text), output_path, "pkl")

    def __save_to_file(self, data: typing.Any, output_path: str, file_format: str, encoding: str = "utf-8") -> None:
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
        :return: None
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

    def __init__(self, dir_name: str, encoding: str | list = "utf-8", file_format: str = "srt") -> None:
        self.__dir_length = len(dir_name) + 1
        self.__file_names = glob(os.path.join(dir_name, f"*.{file_format}"))
        if isinstance(encoding, str):
            self.__encoding = [encoding]
        else:
            self.__encoding = encoding
        self.__subtitles = self.__create_subtitles()
        self.__mfw_size = 150
        self.__mfw = None
        self.__freq_matrix = None
        self.__zscores = None
        self.__delta = None
        self.__kld = None

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

    def get_mfw(self) -> pd.Series:
        return self.__mfw

    def get_freq_matrix(self) -> pd.DataFrame:
        return self.__freq_matrix

    def get_zscores(self) -> pd.DataFrame:
        return self.__zscores

    def get_delta(self) -> pd.DataFrame:
        return self.__delta

    def get_kld(self) -> pd.DataFrame:
        return self.__kld

    def set_mfw_size(self, mfw_size: int) -> None:
        self.__mfw_size = mfw_size

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

    def mfw(self, save: bool = True, output_path: str = "frequency_output") -> typing.Self:
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

        self.__mfw = pd.Series(
            dict(counter),
            index=dict(counter).keys(),
            name="frequency"
        ).sort_values(ascending=False).iloc[:self.__mfw_size]

        if save:
            self.save_to_file(self.__mfw, output_path, "mfw_list_all", "csv")
        return self

    def load_mfw(self) -> typing.Self:
        self.__mfw = self.load_from_file("frequency_output", "mfw_list_all", "csv")
        return self

    def count_tokens(self) -> typing.Self:
        """
        Counts tokens in subtitles based on mfw
        :return: self
        """
        self.__freq_matrix = pd.DataFrame(index=self.__mfw.index)
        for sub in self.__subtitles:
            sub_freq = sub.count_tokens(self.__mfw)
            self.__freq_matrix.insert(0, sub.get_name(), sub_freq)

        return self

    def z_score(self) -> typing.Self:
        self.__zscores = zscore(self.__freq_matrix)
        return self

    def delta(self) -> typing.Self:
        i = 1
        delta_list = []
        cols = self.__zscores.columns
        for col in cols:
            for j in cols[i:]:
                diff = abs(self.__zscores[col] - self.__zscores[j])
                delta_value = diff.sum() / len(self.__zscores)
                delta_list.append([col, j, delta_value])

            i += 1

        self.__delta = pd.DataFrame(delta_list, columns=["sub1", "sub2", "delta"])
        return self

    def burrows_delta(self, save: bool = True, output_path: str = "stylo_out",
                      file_name: str = "burrows_delta") -> typing.Self:
        self.mfw().count_tokens().z_score().delta()
        if save:
            self.save_to_file(
                data=self.__delta,
                output_path=output_path,
                file_name=file_name,
                file_format="csv"
            )
        return self

    def get_corpus_size(self):
        count = 0
        for sub in self.get_text():
            count += len(sub.split())
        return count

    def dirichlet_smoothing(self, tf: int, n: int, mu: float, p_ti_B: float) -> float:
        return (tf / (n * mu)) + (mu / (n * mu)) * p_ti_B

    def kld(self, mu: float):
        self.mfw().count_tokens()
        B = self.get_corpus_size()
        p_t_B = self.__freq_matrix.sum(axis=1) / B

        cols = self.__freq_matrix.columns
        counter = 1

        kld_list = []

        for col_1 in cols:
            for col_2 in cols[counter:]:
                kld_sum = 0
                for i in self.__mfw.index:
                    tf_1 = self.__freq_matrix[col_1][i]
                    n_1 = self.__freq_matrix[col_1].sum()

                    tf_2 = self.__freq_matrix[col_2][i]
                    n_2 = self.__freq_matrix[col_2].sum()

                    p_ti_B = p_t_B.loc[i]

                    p_1 = self.dirichlet_smoothing(tf_1, n_1, mu, p_ti_B)
                    p_2 = self.dirichlet_smoothing(tf_2, n_2, mu, p_ti_B)

                    kld_sum += p_1 * np.log2(p_1 / p_2)
                kld_list.append([col_1, col_2, kld_sum])
            counter += 1

        self.__kld = pd.DataFrame(kld_list, columns=["sub1", "sub2", "kld"])

        self.save_to_file(
            data=self.__kld,
            output_path="stylo_out",
            file_name="kld",
            file_format="csv"
        )

        return self

    def culling(self) -> typing.Self:
        return self

    def cluster(self, trigrams: list) -> typing.Self:
        return self

    def save_to_file(self, data: pd.Series | pd.DataFrame, output_path: str, file_name: str, file_format: str,
                     encoding: str = "utf-8") -> typing.Self:
        path = os.path.join(output_path, f"{file_name}.{file_format}")
        try:
            os.makedirs(output_path)
        except FileExistsError:
            pass

        if file_format == "csv":
            if type(data) == pd.Series:
                data.to_csv(path)
            else:
                data.to_csv(path, index=False)

        return self

    def load_from_file(self, input_path: str, file_name: str, file_format: str) -> typing.Self:
        path = os.path.join(input_path, f"{file_name}.{file_format}")
        if file_format == "csv":
            self.__mfw = pd.read_csv(path)
        return self
