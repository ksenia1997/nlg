import argparse
from utils.csv_process import save_to_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a file, where source input and hypotheses output are joined"
    )

    parser.add_argument('--dataset_source', metavar='PATH', type=str, required=True, help='Path to source file')
    parser.add_argument('--dataset_hypotheses', metavar='PATH', type=str, required=True, help='Path to hypotheses file')
    parser.add_argument('--save_file', metavar='PATH', type=str, required=True,
                        help='Path to file, where result will be saved')

    args = parser.parse_args()

    lines = []
    # args.dataset_source = .data/test.source
    # args.dataset_hypotheses = bart_gpt/test_poetic.hypo
    # save_file = bart_gpt/bart_gpt2_poetic.csv
    with open(args.dataset_source, "r") as source, open(args.dataset_hypotheses, "r") as hypo:
        count = 0
        hypo_data = hypo.readlines()
        for src in source:
            lines.append(src)
            lines.append(hypo_data[count])
            count += 1
            if count == len(hypo_data):
                break
    source.close()
    hypo.close()
    save_to_csv(args.save_file, lines)
