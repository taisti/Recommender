import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split the dataset into n pieces')
    parser.add_argument("--dataset_path", type=str, default="dataset/dataset.csv")
    parser.add_argument("--n_pieces", type=int, default=10,
                        help='Enter the number of pieces you want to divide your dataset into')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    n_pieces = args.n_pieces

    df = pd.read_csv(dataset_path, error_bad_lines=False, encoding='latin-1')

    for idx, df_splitted in enumerate(np.array_split(df, n_pieces)):
        df_splitted.to_csv(f'{dataset_path.split(".")[0]}{idx}.csv')

    print("The dataset was splitted and saved in the directory of the source file")
