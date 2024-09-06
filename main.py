from corpus import *
import time


def main():
    temp0 = time.time()
    # corpus = Corpus("test_corpus_clean", file_format="txt", encoding="utf-8")

    corpus = Corpus("corpus", file_format="srt", encoding="ansi")
    corpus.clean_subtitles()
    temp1 = time.time()
    print("Cleaning done\nDuration:", temp1 - temp0)
    burrows = corpus.burrows_delta().get_delta()
    temp2 = time.time()
    print("Burrows done\nDuration:", temp2 - temp1)
    cluster = corpus.hdbscan("burrows", 3, 3, 0.21, 9, leaf_size=500)
    temp3 = time.time()
    print("HDBSCAN done\nDuration:", temp3 - temp2)
    print(burrows)
    print(cluster)


if __name__ == '__main__':
    main()
