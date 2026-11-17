import argparse
from pathlib import Path
from lifecycle_rec.recommendation.transition_inputs import build_edge_test_data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--transition-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    build_edge_test_data(args.data_dir, args.transition_dir, args.output_dir)
if __name__ == '__main__':
    main()
