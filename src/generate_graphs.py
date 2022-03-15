import argparse
from enum import Enum
from typing import Dict, List, TypedDict

import matplotlib.pyplot as plt

from load_data import ExecutorConfigurationContent, Function, StructureType, load_data


class Metric(Enum):
    DiskBytesSpilled = "diskBytesSpilled"
    ElapsedTime = "elapsedTime"
    ExecutorCpuTime = "executorCpuTime"
    ExecutorRunTime = "executorRunTime"
    PeakExecutionMemory = "peakExecutionMemory"
    ShuffleBytesWritten = "shuffleBytesWritten"
    ShuffleTotalBytesRead = "shuffleTotalBytesRead"

    @property
    def verbose(self):
        return METRIC_VERBOSE[self]


METRIC_VERBOSE: Dict[Metric, str] = {
    Metric.DiskBytesSpilled: "Disk Bytes Spill",
    Metric.ElapsedTime: "Elapsed Time",
    Metric.ExecutorCpuTime: "Executor CPU Time",
    Metric.ExecutorRunTime: "Executor Run Time",
    Metric.PeakExecutionMemory: "Peak Execution Memory",
    Metric.ShuffleBytesWritten: "Shuffle Bytes Written",
    Metric.ShuffleTotalBytesRead: "Shuffle Bytes Read (Total)",
}


class PlotData(TypedDict):
    x: List[int]
    y: List[int]
    err: List[int]


def plot_executors(
    data: ExecutorConfigurationContent,
    function: Function,
):
    for metric in Metric:
        fig, ax = plt.subplots()
        for num_executors, structures in data.items():
            for structure_type, datasets in structures.items():
                output: PlotData = {"x": [], "y": [], "err": []}
                for x in sorted(datasets):
                    output["x"].append(x)
                    output["y"].append(datasets[x][function].mean.get(metric.value))
                    output["err"].append(datasets[x][function].std.get(metric.value))
                ax.errorbar(
                    output["x"],
                    output["y"],
                    yerr=output["err"],
                    label=(
                        f"{function.verbose} - {structure_type.verbose}"
                        f" - {num_executors} executors"
                    ),
                )
        ax.set_title("{} - All data".format(function.verbose))
        ax.set_ylabel("Elapsed time")
        ax.set_xlabel("Dataset size")
        ax.set_xscale("log")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"output/{metric.value}/all-data-{function.value}")
        plt.close(fig)


def plot_comparision_executors(
    data: ExecutorConfigurationContent,
    datasize: int,
    structure_type: StructureType,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
):
    for metric in Metric:
        output: Dict[Function, PlotData] = {}
        for function in functions:
            output[function] = {"x": [], "y": [], "err": []}

        fig, ax = plt.subplots()

        for numExecutor, structures in data.items():
            dataset = structures[structure_type][datasize]
            for function in functions:
                output[function]["x"].append(numExecutor)
                output[function]["y"].append(dataset[function].mean.get(metric.value))
                output[function]["err"].append(dataset[function].std.get(metric.value))

            fig, ax = plt.subplots()

        for function in functions:
            ax.errorbar(
                output[function]["x"],
                output[function]["y"],
                output[function]["err"],
                label=function.verbose,
            )

        ax.set_title(f"{datasize:,} rows - {structure_type.verbose} dataset")
        ax.set_ylabel(metric.verbose)
        ax.set_xlabel("Dataset size")
        ax.set_xscale("log")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            (
                f"output/{metric.value}/{datasize}-rows-"
                f"{structure_type.value}-executor_comparision"
            )
        )
        plt.close(fig)


def plot_comparision(
    data: ExecutorConfigurationContent,
    numExecutors: int,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
):
    for metric in Metric:
        executors = data[numExecutors]
        for structure_type, datasets in executors.items():
            output: Dict[Function, PlotData] = {}
            for function in functions:
                output[function] = {"x": [], "y": [], "err": []}

            for x in sorted(datasets):
                for function in functions:
                    output[function]["x"].append(x)
                    output[function]["y"].append(
                        datasets[x][function].mean.get(metric.value)
                    )
                    output[function]["err"].append(
                        datasets[x][function].std.get(metric.value)
                    )

            fig, ax = plt.subplots()

            for function in functions:
                ax.errorbar(
                    output[function]["x"],
                    output[function]["y"],
                    output[function]["err"],
                    label=function.verbose,
                )

            ax.set_title(f"{numExecutors} executors - {structure_type.verbose}")
            ax.set_ylabel(metric.verbose)
            ax.set_xlabel("Dataset size")
            ax.set_xscale("log")
            ax.legend()
            plt.tight_layout()
            functions_string = "_".join([function.value for function in functions])
            plt.savefig(
                (
                    f"output/{metric.value}/{numExecutors}-executors-"
                    f"{structure_type.value}-{functions_string}"
                )
            )
            plt.close(fig)


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
    plot_comparision(data, 2)
    plot_executors(data, Function.EarlyStopSortMerge)
    plot_executors(data, Function.Union)
    plot_executors(data, Function.Exploding)

    plot_comparision(data, 2)
    plot_executors(data, Function.EarlyStopSortMerge)
    plot_executors(data, Function.Union)
    plot_executors(data, Function.Exploding)

    for structure_type in StructureType:
        plot_comparision_executors(data, 10_000_000, structure_type)
