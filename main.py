from corpus import *


def test():
    """
    Test function to test diffrent parameters on the given dataset
    :return: None
    """
    mfw_list = [150, 200, 300, 400, 500, 700, 1000]
    mu_list = [0.1, 0.3, 0.5, 1, 3, 5, 10, 20, 25]
    corpus = Corpus("test_run", file_format="srt", encoding="ansi")
    corpus.clean_subtitles()

    corpus.ner()
    corpus.set_ner_usage(True)

    # Test Burrows mit NER
    for mfw in mfw_list:
        corpus.set_mfw_size(mfw)
        corpus.burrows_delta(save=False).gephi_input(output_path="stylo_out/test_run", file_name=f"burrows_{mfw}_ner")

    # Test KLD mit NER
    for mfw in mfw_list:
        corpus.set_mfw_size(mfw)
        for mu in mu_list:
            corpus.kld(mu, save=False).gephi_input(output_path="stylo_out/test_run", file_name=f"kld_{mfw}_{mu}_ner")

    # Test Labbe mit NER
    corpus.labbe(save=False).gephi_input(output_path="stylo_out/test_run", file_name="labbe_ner")


def main():
    """
    Main function used for calculating the distances of the complete corpus
    :return: None
    """
    corpus = Corpus("corpus", "latin_1", "srt")
    corpus.ner()
    corpus.set_ner_usage(True)
    corpus.set_mfw_size(1000)
    corpus.kld(10, save=False).gephi_input(file_name="corpus")


def demo():
    """
    Demo function to run an example of each distance measure
    :return: None
    """
    corpus = Corpus("test_run", "latin_1", "srt")
    corpus.burrows_delta(save=False).gephi_input(output_path="demo", file_name="demo_run_burrows")
    corpus.kld(10, save=False).gephi_input(output_path="demo", file_name="demo_run_kld")
    corpus.labbe(save=False).gephi_input(output_path="demo", file_name="demo_run_labbe")


if __name__ == '__main__':
    demo()
