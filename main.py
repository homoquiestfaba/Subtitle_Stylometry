from corpus import *


def test():
    mfw_list = [150, 200, 300, 400, 500, 700, 1000]
    mu_list = [0.1, 0.3, 0.5, 1, 3, 5, 10, 20, 25]
    corpus = Corpus("test_run", file_format="srt", encoding="ansi")
    corpus.clean_subtitles()

    # Test Burrows
    for mfw in mfw_list:
        corpus.set_mfw_size(mfw)
        corpus.burrows_delta(save=False).gephi_input(output_path="stylo_out", file_name=f"burrows_{mfw}")

    # Test KLD
    for mfw in mfw_list:
        corpus.set_mfw_size(mfw)
        for mu in mu_list:
            corpus.kld(mu, save=False).gephi_input(output_path="stylo_out", file_name=f"kld_{mfw}_{mu}")

    # Test Labbe
    corpus.labbe(save=False).gephi_input(output_path="stylo_out", file_name=f"labbe")

    corpus.ner()
    corpus.set_ner_usage(True)

    # Test Burrows mit NER
    for mfw in mfw_list:
        corpus.set_mfw_size(mfw)
        corpus.burrows_delta(save=False).gephi_input(output_path="stylo_out", file_name=f"burrows_{mfw}_ner")

    # Test KLD mit NER
    for mfw in mfw_list:
        corpus.set_mfw_size(mfw)
        for mu in mu_list:
            corpus.kld(mu, save=False).gephi_input(output_path="stylo_out", file_name=f"kld_{mfw}_{mu}_ner")

    # Test Labbe mit NER
    corpus.labbe(save=False).gephi_input(output_path="stylo_out", file_name=f"labbe_ner")


if __name__ == '__main__':
    test()
