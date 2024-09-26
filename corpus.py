"""
This file contains six classes that create a Corpus- and Subtitle-object and Distance objects. The Corpus-object is to
be called to operate on the subtitle corpus. It automatically creates a Subtitle-object for every subtitle file in the
corpus and handles them through its methods.
The Distance objects handle the output as distance matrix or Gephi input
"""
from glob import glob
import typing
import re
import pickle
import os
import tqdm
import collections

import spacy
import pandas as pd
import numpy as np
from numpy import dtype
from scipy.stats import zscore
from scipy.special import binom
import torch

import warnings

warnings.filterwarnings("ignore")

# Set DEVICE to CPU or GPU
DEVICE = "CPU"


class Subtitle:
    """
    The Subtitle-class represents the subtitles of a single movie. It presents multiple methods for operating on a
    subtitle file and saving them for later processing to reduce process time once you've done it.
    """
    timestamps = re.compile(r"(\d+\s*)?\d\d:\d\d:\d\d[,\.]\d\d\d|-->")
    punctuation = re.compile(r"[!\"#$%&\'()*+,./:;<=>?@\[\]^_`{|}~\\-]")
    spaces = re.compile(r"\s\s+")
    brackets_1 = re.compile(r"<.*>")
    brackets_2 = re.compile(r"\(.*\)")

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

    def clean(self, save: bool, output_path: str, encoding: str) -> None:
        """
        Cleans SRT-file by removing SRT-syntactic metadata (e.g. time to display text) and added information in brackets
        :return: None
        """
        self.__text = self.timestamps.sub("", self.__text)
        self.__text = self.punctuation.sub("", self.__text)
        self.__text = self.spaces.sub(" ", self.__text)
        self.__text = self.brackets_1.sub("", self.__text)
        self.__text = self.brackets_2.sub("", self.__text)
        if save:
            self.__save_to_file(self.__text, output_path, "txt", encoding)

    def ner(self, nlp, save: bool, output_path: str) -> None:
        """
        Performs NER with SpaCy-Model "en_core_web_trf" and replaces words with entity type
        Example: "Sauron betrayed them all" --> "PERSON betrayed them all"]
        :param nlp: spacy model
        :param save: Whether the file is saved as pickle
        :param output_path: Output folder for saving file
        :return: None
        """
        doc = nlp(self.__text)
        self.__ner_text = " ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc])
        if save:
            self.__save_to_file(self.__text, output_path, "pkl")

    def count_tokens(self, mfw: pd.Series, ner_usage: bool, freqs: dict) -> pd.Series:
        """
        Counts all tokens of the subtitle file that are in the most frequent word list
        :param freqs: vocabulary dictionary
        :param ner_usage: whether to count NER text or plain text
        :param mfw: most frequent word list from whole corpus
        :return: pd.Series of token frequencies
        """
        if ner_usage:
            text = self.__ner_text
        else:
            text = self.__text
        total_freq = dict(collections.Counter(text.split()))
        for term in total_freq.keys():
            if term in freqs.keys():
                freqs[term] = total_freq[term]

        return pd.Series(freqs.values(), index=mfw.index)

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


class Distance:
    """
    Superclass of all distances (Burrows, Kld, Labbe) that generates a distance matrix and output for Gephi
    """

    def __init__(self, distance: pd.DataFrame, names: list, dist_type: str):
        self.__distance = distance
        self.__name_list = names
        self.__dist_matrix = None
        self.__pruned = None
        self.__dist_type = dist_type

    def get_dist_matrix(self) -> pd.DataFrame:
        return self.__dist_matrix

    def get_pruned(self) -> pd.DataFrame:
        return self.__pruned

    def make_dist_matrix(self, output_path: str = None, file_name: str = None, save: bool = False) -> typing.Self:
        """
        Claculates distance matrix from distance list
        :param output_path: path to output directory
        :param file_name: output file name
        :param save: whether to save or not
        :return: self
        """
        names = self.__name_list
        dist_matrix = []
        cout = 0

        data = self.__distance
        print("Data loaded to Distance Class")

        if DEVICE == "GPU":
            dist_tensor = torch.zeros(len(names), len(names), device="cuda")

            i = [i for i in range(len(names))]
            sub_dict = dict(zip(sorted(names), i))
            print("SubDict")

            sub1 = [sub_dict[sub] for sub in data["sub1"]]
            sub2 = [sub_dict[sub] for sub in data["sub2"]]
            print("Sub lists")

            indices_1 = torch.tensor(tuple(zip(sub1, sub2)), device="cuda")
            indices_2 = torch.tensor(tuple(zip(sub2, sub1)), device="cuda")
            values = data[self.__dist_type]
            values = torch.tensor(values, device="cuda").float()
            print("I and V")

            dist_tensor[indices_1[:, 0], indices_1[:, 1]] = values
            print("Tensor_1")
            dist_tensor[indices_2[:, 0], indices_2[:, 1]] = values
            print("Tensor_2")

            self.__dist_matrix = pd.DataFrame(dist_tensor.cpu().numpy(), columns=sub_dict.keys(), index=sub_dict.keys())
            print("DataFrame")

        elif DEVICE == "CPU":
            for name in tqdm.tqdm(names, desc="Creating Dist-Matrix"):
                sub_dists = pd.concat([data[data.sub1 == name], data[data.sub2 == name]])
                sub_partners_1 = sub_dists[sub_dists.sub1 != name]["sub1"].tolist()
                sub_partners_2 = sub_dists[sub_dists.sub2 != name]["sub2"].tolist()
                sub_partners = sub_partners_2 + sub_partners_1 + [name]

                values = sub_dists[self.__dist_type].tolist()
                values.append(0)

                dist_matrix.append(pd.Series(values, index=sub_partners).sort_index().values)

            self.__dist_matrix = pd.DataFrame(dist_matrix, columns=names, index=names).astype("float32")

        if save:
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass
            self.__dist_matrix.to_csv(f"{output_path}/{file_name}.csv", index=False)
        # output.astype(float)
        return self

    def prune(self, inverse: bool = True, n_nearest: int = 3, output_path: str = "gephi_in",
              file_name: str = "pruned") -> None:
        """
        Creates Gephi input by selecting only most similar films
        :param inverse: inverse distances for network weights
        :param n_nearest: number of nearest films to keep
        :param output_path: path to output directory
        :param file_name: output file name
        :return: None
        """
        output = self.__dist_matrix
        pruned_output = []
        for i in tqdm.tqdm(output.index, desc="Pruning"):
            m = output.loc[i].astype(float).nsmallest(n_nearest + 1).index.tolist()
            for j in m:
                if output[j][i] == 0:
                    continue
                if inverse:
                    pruned_output.append([i, j, round(1 / output[j][i], 6)])
                else:
                    pruned_output.append([i, j, output[j][i]])

        self.__pruned = pd.DataFrame(pruned_output, columns=["Source", "Target", "Weight"])

        try:
            os.makedirs(output_path)
        except FileExistsError:
            pass
        self.__pruned.to_csv(f"{output_path}/{file_name}.csv", index=False)

    def gephi_input(self, inverse: bool = True, n_nearest: int = 3, output_path: str = "gephi_in",
                    file_name: str = "gephi_in") -> None:
        """
        Creates Gephi input by using the methods make_dist_matrix and prune
        :param inverse: like prune
        :param n_nearest: like prune
        :param output_path: like prune
        :param file_name: like prune
        :return: None
        """
        self.make_dist_matrix().prune(inverse, n_nearest, output_path, file_name)


class Burrows(Distance):
    """
    Class for distances calculated with Burrows' Delta
    """

    def __init__(self, delta: pd.DataFrame, names: list):
        super().__init__(delta, names, "delta")

    def get_delta(self) -> pd.DataFrame:
        return self.__distance


class Kld(Distance):
    """
    Class for distances calculated by the Kullback-Leibler-Divergence
    """

    def __init__(self, kld: pd.DataFrame, names: list, mu: float):
        super().__init__(kld, names, "kld")
        self.__mu = mu

    def get_kld(self) -> pd.DataFrame:
        return self.__distance

    def get_mu(self) -> float:
        return self.__mu


class Labbe(Distance):
    """
    Class for distances calculated by the Labbe distance
    """

    def __init__(self, labbe: pd.DataFrame, names: list):
        super().__init__(labbe, names, "labbe")

    def get_labbe(self) -> pd.DataFrame:
        return self.__distance


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
        self.__name_list = sorted(
            [
                sub.get_name()
                for sub in self.__subtitles
            ]
        )
        self.__mfw_size = 150
        self.__ner_usage = False
        self.__mfw = None
        self.__freq_matrix = None
        self.__zscores = None

    def __create_subtitles(self) -> typing.List[Subtitle]:
        """
        Creates Subtitle-objects for every subtitle in corpus
        :return: list of Subtitle-Objects
        """
        return [Subtitle(
            file_name=name,
            dir_length=self.__dir_length,
            encoding=self.__encoding)
            for name in tqdm.tqdm(self.__file_names, desc="Creating Corpus")
        ]

    def get_subtitles(self) -> typing.List[Subtitle]:
        return self.__subtitles

    def get_name_list(self) -> typing.List[str]:
        return self.__name_list

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

    def get_corpus_size(self):
        count = 0
        for sub in self.get_text():
            count += len(sub.split())
        return count

    def get_corpus_vocabulary(self):
        token_list = []
        for sub in self.get_text():
            token_list.extend(sub.split())

        token_dict = dict(collections.Counter(token_list))

        return pd.Series(token_dict, index=token_dict.keys()).sort_values(ascending=False)

    def set_mfw_size(self, mfw_size: int) -> None:
        self.__mfw_size = mfw_size

    def set_ner_usage(self, usage: bool) -> None:
        self.__ner_usage = usage

    def clean_subtitles(self, save: bool = False, output_path: str = "cleaned_output",
                        output_encoding: str = "utf-8") -> typing.Self:
        """
        Cleans whole corpus of subtitle files and writes it to txt file
        :param save: Whether the file is to be saved
        :param output_path: folder to write cleaned files
        :param output_encoding: encoding of txt output file
        :return: None
        """
        for subtitle in tqdm.tqdm(self.__subtitles, desc="Cleaning Subtitles"):
            subtitle.clean(save=save,
                           output_path=output_path,
                           encoding=output_encoding)
        return self

    def load_cleaned_subtitle(self, input_path: str = "cleaned_output") -> typing.Self:
        """
        Loads saved cleaned subtitle file
        :param input_path: path name for file
        :return: self
        """
        for sub in self.__subtitles:
            sub.load_from_file(input_path=input_path, mode="txt")
        return self

    def ner(self, save: bool = True, output_path: str = "ner_output") -> typing.Self:
        """
        Performs NER on whole corpus and replaces entities with generic names
        Optional: Save results to default ("ner_output/") or user defined directory
        :param save: Whether the file is saved as pickle
        :param output_path: path to output directory in case of saving
        :return: self
        """
        i = 1
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("merge_entities")
        for sub in tqdm.tqdm(self.__subtitles, desc="NER"):
            # print(str(i) + "/" + str(len(self.__subtitles)))
            sub.ner(
                nlp=nlp,
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
        for sub in tqdm.tqdm(self.__subtitles, desc="Loading NER"):
            sub.load_from_file(input_path=input_path, mode="ner")
        return self

    def mfw(self, save: bool = True, output_path: str = "frequency_output") -> typing.Self:
        """
        Counts frequencies of words in corpus
        :param save: whether Data should be saved. Default: True
        :param output_path: output directory to save frequencies list
        :return: self
        """
        if not self.__ner_usage:
            counter = collections.Counter()
            [
                counter.update(
                    sub.get_plain_text().split()
                )
                for sub in self.__subtitles
            ]
        else:
            counter = collections.Counter()
            [
                counter.update(
                    sub.get_ner_text().split()
                )
                for sub in self.__subtitles
            ]

        self.__mfw = pd.Series(
            dict(counter),
            index=dict(counter).keys(),
            name="frequency"
        ).sort_values(ascending=False).iloc[:self.__mfw_size]

        if save:
            self.__save_to_file(self.__mfw, output_path, "mfw_list_all", "csv")
        return self

    def load_mfw(self, input_path: str = "frequency_output", file_name: str = "mfw_list_all") -> typing.Self:
        """
        Loads mfw-file for corpus
        :param input_path: path name for file
        :param file_name: name of file to load
        :return:
        """
        self.__mfw = self.load_from_file(input_path, file_name, "csv")
        return self

    def count_mfw_tokens(self, save: bool = True, output_path: str = "frequency_output") -> typing.Self:
        """
        Counts tokens in subtitles based on mfw
        :return: self
        """
        # self.__freq_matrix = pd.DataFrame()
        sub_counts = []
        sub_names = []
        freqs = dict(zip(self.__mfw.index, [0] * self.__mfw_size))
        for sub in self.__subtitles:
            temp_freqs = freqs.copy()
            sub_freq = sub.count_tokens(self.__mfw, self.__ner_usage, temp_freqs)
            # self.__freq_matrix.insert(0, sub.get_name(), sub_freq)
            sub_counts.append(sub_freq)
            sub_names.append(sub.get_name())
        self.__freq_matrix = pd.concat(sub_counts, axis=1)
        self.__freq_matrix = pd.DataFrame(self.__freq_matrix.values, columns=sub_names)

        if save:
            self.__save_to_file(self.__freq_matrix, output_path, "mfw_frequency_table", "csv")

        return self

    def count_all_tokens(self) -> typing.Self:
        """
        Counts every token in subtitles
        :return: self
        """
        vocabulary = self.get_corpus_vocabulary()
        self.__freq_matrix = pd.DataFrame(index=vocabulary.index)
        freqs = dict(zip(vocabulary.index, [0] * len(vocabulary.index)))
        for sub in tqdm.tqdm(self.__subtitles):
            temp_freqs = freqs.copy()
            sub_freq = sub.count_tokens(vocabulary, self.__ner_usage, temp_freqs)
            self.__freq_matrix.insert(0, sub.get_name(), sub_freq)

        return self

    def z_score(self) -> typing.Self:
        """
        Calculates Z-Score of frequencies
        :return: self
        """
        self.__zscores = zscore(self.__freq_matrix)
        return self

    def delta(self) -> Burrows:
        """
        Calculates Burrows' Delta
        :return: self
        """
        delta_list = []

        cols = self.__zscores.columns

        if DEVICE == "GPU":
            data = torch.tensor(self.__zscores.T.values, device="cuda")

            for i, col1 in enumerate(tqdm.tqdm(cols, desc="Burrows' Delta")):
                j = i + 1

                delta_values = torch.nn.PairwiseDistance(p=1.0)(
                    data[i,], data[j:, ],
                ) / len(self.__zscores)

                for k, col2 in enumerate(cols[j:]):
                    delta_list.append([col1, col2, round(delta_values[k].item(), 6)])

                # diff = abs(self.__zscores[col] - self.__zscores[j])
                # delta_value = diff.sum() / len(self.__zscores)

            delta = pd.DataFrame(delta_list, columns=["sub1", "sub2", "delta"])

        elif DEVICE == "CPU":
            cout = 0
            i = 1
            delta_list = np.empty((int(binom(len(self.__name_list), 2)), 3), dtype)

            for col in tqdm.tqdm(cols):
                for j in cols[i:]:
                    diff = abs(self.__zscores[col] - self.__zscores[j])
                    delta_value = diff.sum() / len(self.__zscores)

                    delta_list[cout] = [col, j, round(delta_value, 6)]

                    cout += 1
                    if cout % 10000 == 0:
                        print(cout)
                i += 1

            delta = pd.DataFrame(delta_list, columns=["sub1", "sub2", "delta"])

        else:
            raise ValueError("Device must be either GPU or CPU")

        return Burrows(delta, self.__name_list)

    def burrows_delta(self, save: bool = True, output_path: str = "stylo_out",
                      file_name: str = "burrows_delta") -> Burrows:
        """
        Collects methods to calculate Burrows' Delta
        :param save: whether to save Burrows' Delta
        :param output_path: path to save Burrows' Delta'
        :param file_name: name of output file
        :return: Burrows
        """
        delta = self.mfw().count_mfw_tokens().z_score().delta()

        if save:
            self.__save_to_file(data=delta, output_path=output_path, file_name=file_name, file_format="csv")
        return delta

    def kld(self, mu: float, save: bool = True, output_path: str = "stylo_out", file_name: str = "kld") -> Kld:
        """
        Claculates Kullback-Leibler-Divergence
        :param mu: parameter for Dirichlet smoothing
        :param save: whether to save KLD
        :param output_path: path to save KLD
        :param file_name: name of output file
        :return: Kld
        """
        self.mfw().count_mfw_tokens()
        B = self.get_corpus_size()

        cols = self.__freq_matrix.columns

        kld_list = []
        if DEVICE == "CPU":
            p_t_B = self.__freq_matrix.sum(axis=1) / B
            counter = 1
            for col_1 in tqdm.tqdm(cols):
                for col_2 in cols[counter:]:
                    tf_1 = self.__freq_matrix[col_1]
                    n_1 = self.__freq_matrix[col_1].sum()

                    tf_2 = self.__freq_matrix[col_2]
                    n_2 = self.__freq_matrix[col_2].sum()

                    p_1 = (tf_1 / (n_1 * mu)) + (mu / (n_1 * mu)) * p_t_B
                    p_2 = (tf_2 / (n_2 * mu)) + (mu / (n_2 * mu)) * p_t_B

                    kld_sum = p_1 * np.log2(p_1 / p_2)

                    kld_list.append([col_1, col_2, round(kld_sum.sum(), 6)])
                counter += 1

        elif DEVICE == "GPU":
            p_t_B = torch.tensor((self.__freq_matrix.sum(axis=1) / B).T.values, device="cuda")
            data = torch.tensor(self.__freq_matrix.T.values, device="cuda")

            for i, col1 in enumerate(tqdm.tqdm(cols, desc="KLD")):
                j = i + 1

                tf_1 = data[i,]
                n_1 = data[i,].sum()

                tf_2 = data[j:, ]
                n_2 = torch.tile(data[j:, ].sum(dim=1), (self.__mfw_size, 1)).T

                p_1 = (tf_1 / (n_1 * mu)) + (mu / (n_1 * mu)) * p_t_B
                p_2 = (tf_2 / (n_2 * mu)) + (mu / (n_2 * mu)) * p_t_B

                kld_values = (p_1 * torch.log2(p_1 / p_2)).sum(dim=1)

                for k, col2 in enumerate(cols[j:]):
                    kld_list.append([col1, col2, round(kld_values[k].item(), 6)])

        else:
            raise ValueError("Device must be either GPU or CPU")

        kld = pd.DataFrame(kld_list, columns=["sub1", "sub2", "kld"])

        if save:
            self.__save_to_file(data=kld, output_path=output_path, file_name=file_name, file_format="csv")

        return Kld(kld, self.__name_list, mu)

    def labbe(self, save: bool = True, output_path: str = "stylo_out", file_name: str = "labbe") -> Labbe:
        """
        Calculates Labbé's intertextual distance
        :param save: whether to save labbe
        :param output_path: path to save labbe
        :param file_name: name of output file
        :return: Labbe
        """
        self.count_all_tokens()
        cols = self.__freq_matrix.columns
        counter = 1

        labbe_list = []
        if DEVICE == "CPU":
            for col_1 in tqdm.tqdm(cols):
                na = self.__freq_matrix[col_1].sum()
                for col_2 in cols[counter:]:
                    nq = self.__freq_matrix[col_2].sum()
                    tfi_a = self.__freq_matrix[col_1]
                    tfi_q = self.__freq_matrix[col_2]
                    over_sum = abs((tfi_a * (nq / na)) - tfi_q)
                    d_labbe = over_sum.sum() / (2 * nq)
                    labbe_list.append([col_1, col_2, round(d_labbe, 6)])
                counter += 1

        elif DEVICE == "GPU":
            data = torch.tensor(self.__freq_matrix.T.values, device="cuda")

            for i, col1 in enumerate(tqdm.tqdm(cols, desc="Labbe")):
                j = i + 1

                size = data[i,].size()[0]

                tfi_a = data[i,]
                tfi_q = data[j:, ]

                na = data[i,].sum()
                nq = torch.tile(data[j:, ].sum(dim=1), (size, 1)).T
                nq_divide = data[j:, ].sum(dim=1)

                tfi_a_norm = tfi_a * (nq / na)

                labbe_values = torch.nn.PairwiseDistance(p=1.0)(
                    tfi_a_norm, tfi_q
                )

                labbe_values = labbe_values / (2 * nq_divide)

                for k, col2 in enumerate(cols[j:]):
                    labbe_list.append([col1, col2, round(labbe_values[k].item(), 6)])
        else:
            raise ValueError("Device must be either GPU or CPU")

        labbe = Labbe(pd.DataFrame(labbe_list, columns=["sub1", "sub2", "labbe"]), self.__name_list)

        if save:
            self.__save_to_file(data=labbe, output_path=output_path, file_name=file_name, file_format="csv")

        return labbe

    def __save_to_file(self, data: pd.Series | pd.DataFrame | dict, output_path: str, file_name: str, file_format: str,
                       encoding: str = "utf-8") -> typing.Self:
        """
        Method to handle saving internally
        :param data: data to be saved
        :param output_path: path to save data
        :param file_name: name of output file
        :param file_format: format of output file
        :param encoding: encoding of output file. Default is UTF-8
        :return: self
        """
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
        """
        Method to handle loading files
        :param input_path: path of file to load
        :param file_name: name of input file
        :param file_format: format of input file
        :return: self
        """
        path = os.path.join(input_path, f"{file_name}.{file_format}")
        if file_format == "csv":
            self.__mfw = pd.read_csv(path)
        return self
