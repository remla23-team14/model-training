"""
Get data from a source and dump it for use
"""

import os

import pandas as pd

# current_directory = os.path.dirname(__file__)
# parent_directory = os.path.split(current_directory)[0]


def read_data(f_path: str) -> pd.DataFrame:
    """Read data from csv file"""
    # dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)
    dataset = pd.read_csv(
        f_path, delimiter="\t", quoting=3, dtype={"Review": object, "Liked": int}
    )

    dataset = dataset[["Review", "Liked"]]
    return dataset


def write_data(dataset: pd.DataFrame, output_path: str) -> None:
    """Dump to csv as raw data"""
    dataset.to_csv(output_path, index=False)


def main() -> None:
    """Read data and write to file"""
    f_path = os.path.join("data", "external", "a1_RestaurantReviews_HistoricDump.tsv")
    output_path = os.path.join("data", "raw", "train_dataset.csv")
    dataset = read_data(f_path)

    print(dataset.shape)
    print(dataset.head())
    write_data(dataset, output_path)


if __name__ == "__main__":
    main()
