from utils.csv_process import save_to_csv


def join_test_source_and_hypo():
    lines = []
    with open('.data/test.source', "r") as source, open('mixed_bart/test_combined.hypo', "r") as hypo:
        count = 0
        hypo_data = hypo.readlines()
        for src in source:
            lines.append(src)
            lines.append(hypo_data[count])
            count += 1
            if count == 21:
                break
    source.close()
    hypo.close()
    save_to_csv("mixed_bart/bart_gpt2_7_3.csv", lines)


join_test_source_and_hypo()
