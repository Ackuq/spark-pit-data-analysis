import argparse
from typing import List, cast

import pandas as pd

from generate_graphs import Metric
from load_data import ExecutorConfigurationContent, load_data


def se(data: ExecutorConfigurationContent):
    # Percentage SE
    for metric in Metric:
        all_data: List[List] = []
        for num_executors, executor in data.items():
            for structure_type, structure in executor.items():
                for dataset_size, function in structure.items():
                    for function_type, dataset in function.items():
                        mean = cast(float, dataset.mean.get(metric.value))
                        std = cast(float, dataset.std.get(metric.value))
                        std_percentage = std / mean if not mean == 0.0 else 0.0
                        all_data.append(
                            [
                                num_executors,
                                structure_type.value,
                                function_type.value,
                                dataset_size,
                                std_percentage,
                            ]
                        )
        pd.DataFrame(
            all_data,
            columns=[
                "executors",
                "structure_type",
                "function",
                "dataset_size",
                "std_percentage",
            ],
        ).to_csv(f"./std/{metric.value}.csv")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog="Graph generation")

    argument_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./results",
        help="The location of the input dircectory",
    )

    args = argument_parser.parse_args()

    data = load_data(args.input)
    se(data)
