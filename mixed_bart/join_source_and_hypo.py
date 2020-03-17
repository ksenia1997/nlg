from utils.csv_process import save_to_csv


def join_test_source_and_hypo():
    lines = []
    with open('.data/test.source', "r") as source, open('mixed_bart/test_tfidf.hypo', "r") as hypo:
        count = 0
        hypo_data = hypo.readlines()
        for src in source:
            lines.append(src)
            lines.append(hypo_data[count])
            count += 1
    source.close()
    hypo.close()
    save_to_csv("mixed_bart/bart_result__tfidf.csv", lines)


join_test_source_and_hypo()
