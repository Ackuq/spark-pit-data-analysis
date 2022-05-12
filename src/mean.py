import argparse
from typing import Dict, List

import pandas as pd
from numpy import float_, int_
from scipy import stats

from generate_graphs import Metric
from load_data import ExecutorConfigurationContent, Function, StructureType, load_data

metrics = [Metric.ElapsedTime, Metric.PeakExecutionMemory]


def save_speedup(
    data: ExecutorConfigurationContent,
    structures: List[StructureType] = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
        # StructureType.AscendingBucketed20,
        # StructureType.AscendingBucketed40,
        # StructureType.AscendingBucketed80,
        # StructureType.AscendingBucketed160,
    ],
):
    num_executors = sorted(list(data.keys()))
    baseline = num_executors.pop(0)
    for structure_type in structures:
        for function in Function:
            for executors in num_executors:
                dataframe = pd.DataFrame()
                structure = data[executors][structure_type]
                for dataset_size, dataset in structure.items():
                    df = pd.DataFrame(
                        [
                            data[baseline][structure_type][dataset_size][
                                function
                            ].mean.get(Metric.ElapsedTime.value, 0.0)
                            / dataset[function].mean.get(Metric.ElapsedTime.value, 0.0)
                        ]
                    )
                    df = df.assign(size=[dataset_size])
                    dataframe = pd.concat([dataframe, df])

                dataframe.sort_values("size").to_csv(
                    "./speedup/{}-{}-{}_executors.csv".format(
                        structure_type.value, function.value, executors
                    )
                )


def save_mean(
    data: ExecutorConfigurationContent,
    structures: List[StructureType] = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
        # StructureType.AscendingBucketed20,
        # StructureType.AscendingBucketed40,
        # StructureType.AscendingBucketed80,
        # StructureType.AscendingBucketed160,
    ],
):
    executors = data[2]
    linear_regression = pd.DataFrame()
    for structure_type in sorted(structures):
        structure = executors[structure_type]
        means: Dict[Function, pd.DataFrame] = {
            Function.Exploding: pd.DataFrame(),
            Function.EarlyStopSortMerge: pd.DataFrame(),
            Function.Union: pd.DataFrame(),
        }
        for dataset_size, dataset in structure.items():
            for function in Function:
                dataframe = pd.DataFrame(
                    [dataset[function].mean, dataset[function].std]
                )
                dataframe = dataframe[[metric.value for metric in metrics]]
                dataframe = dataframe.assign(type=["mean", "std"])
                dataframe_with_size = dataframe.assign(
                    size=[dataset_size, dataset_size]
                )

                means[function] = pd.concat(
                    [means[function], dataframe_with_size.iloc[0].to_frame()], axis=1
                )
                dataframe.to_csv(
                    "./mean/2-executors/{}-{}-{}.csv".format(
                        structure_type.value, dataset_size, function.value
                    )
                )

        for function, df in means.items():
            for metric in metrics:
                values = stats.linregress(
                    df.loc["size"].to_numpy().astype(int_),
                    df.loc[metric.value].to_numpy().astype(float_),
                )
                values_df = (
                    pd.DataFrame(
                        [values],
                        columns=["slope", "intercept", "rvalue", "pvalue", "stderr"],
                    )
                    .assign(function=[function.name])
                    .assign(dataStructure=structure_type.value)
                    .assign(metric=metric.value)
                )
                linear_regression = pd.concat([linear_regression, values_df])

    linear_regression.to_csv("./regression_analysis.csv", float_format="%12.3e")


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
    save_mean(data)
    save_speedup(data)
