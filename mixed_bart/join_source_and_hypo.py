from utils.csv_process import save_to_csv


def join_test_source_and_hypo():
    lines = []
    with open('../datasets/test.source', "r") as source, open('test.hypo', "r") as hypo:
        count = 0
        hypo_data = hypo.readlines()
        for src in source:
            lines.append(src)
            lines.append(hypo_data[count])
            count += 1
    source.close()
    hypo.close()
    save_to_csv("bart_result.csv", lines)


join_test_source_and_hypo()
