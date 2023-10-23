import argparse
from typing import List

import pandas as pd


class DataGenerator:
    def __init__(self, args: argparse.Namespace, connector=None):
        self.args = args
        if connector:
            print("doing nothing")

    def get_data(self, csv_path) -> pd.DataFrame:
        with open(csv_path) as f:
            self.df = pd.read_csv(f)

        if self.args.num_samples:
            self.df = self.df.sample(self.args.num_samples)

    def run(self) -> None:
        for row in self.df.iterrows():
            print(f"Doing some preprocessing here on row {row}")

    def write_to_csv(self, path) -> None:
        self.df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_generator = DataGenerator(args)
    data_generator.run()
    data_generator.write_to_csv()
