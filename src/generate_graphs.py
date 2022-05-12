import argparse
import os
from enum import Enum
from typing import Dict, List, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from load_data import (
    DatasetContent,
    ExecutorConfigurationContent,
    Function,
    HiveContent,
    StructureType,
    StructureTypeContent,
    load_data,
)


class Metric(Enum):
    DiskBytesSpilled = "diskBytesSpilled"
    ElapsedTime = "elapsedTime"
    ExecutorCpuTime = "executorCpuTime"
    ExecutorRunTime = "executorRunTime"
    PeakExecutionMemory = "peakExecutionMemory"
    ShuffleBytesWritten = "shuffleBytesWritten"
    ShuffleTotalBytesRead = "shuffleTotalBytesRead"
    MemoryBytesSpilled = "memoryBytesSpilled"
    ShuffleFetchWaitTime = "shuffleFetchWaitTime"
    ShuffleWriteTime = "shuffleWriteTime"

    @property
    def verbose(self):
        return METRIC_VERBOSE[self]


METRIC_VERBOSE: Dict[Metric, str] = {
    Metric.DiskBytesSpilled: "Disk Bytes Spill (Bytes)",
    Metric.ElapsedTime: "Elapsed Time (ms)",
    Metric.ExecutorCpuTime: "Executor CPU Time (ms)",
    Metric.ExecutorRunTime: "Executor Run Time (ms)",
    Metric.PeakExecutionMemory: "Peak Execution Memory (Bytes)",
    Metric.ShuffleBytesWritten: "Shuffle Bytes Written (Bytes)",
    Metric.ShuffleTotalBytesRead: "Shuffle Bytes Read (Bytes)",
    Metric.MemoryBytesSpilled: "Memory Bytes Spill (Bytes)",
    Metric.ShuffleFetchWaitTime: "Shuffle Fetch Wait Time (ms)",
    Metric.ShuffleWriteTime: "Shuffle Write Time (ms)",
}


class PlotData(TypedDict):
    x: List[int]
    y: List[Union[int, float]]
    err: List[Union[int, float]]


LS_MAP = {Function.EarlyStopSortMerge: "solid", Function.Exploding: "dashed", Function.Union: "dashdot"}

CombinedStructures = List[Tuple[List[StructureType], DatasetContent]]


def _combine_structures(
    executors: StructureTypeContent,
    structures: List[StructureType],
    function: Function,
    metric: Metric,
) -> CombinedStructures:
    combined_structures: CombinedStructures = []
    for structure_type in sorted(structures):
        found = False
        for other_structure in combined_structures:
            # Check if content is the same
            current = executors[structure_type]
            other = other_structure[1]
            if set(current.keys()) == set(other.keys()):
                for x in sorted(current):
                    current_value = current[x][function].mean.get(metric.value)
                    other_value = other[x][function].mean.get(metric.value)
                    if (current_value == 0.0 and other_value == 0.0) or (
                        current_value != 0.0
                        and abs(current_value - other_value) / current_value  # type: ignore
                        # Differ by at most 0.2%
                    ) <= 0.002:
                        found = True
                        other_structure[0].append(structure_type)
                        break
        if not found:
            combined_structures.append(([structure_type], executors[structure_type]))
    return combined_structures


def plot_executors(
    data: ExecutorConfigurationContent,
    function: Function,
    structures: List[StructureType] = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
        StructureType.AscendingBucketed20,
        StructureType.AscendingBucketed40,
        StructureType.AscendingBucketed80,
        StructureType.AscendingBucketed160,
    ],
):
    num_executors = sorted(list(data.keys()))
    for metric in Metric:
        figs: List[Tuple[Figure, str]] = []
        for structure_type in structures:
            fig, ax = plt.subplots()
            for executor in num_executors:
                output: PlotData = {"x": [], "y": [], "err": []}
                datasets = data[executor][structure_type]
                for x in sorted(datasets):
                    output["x"].append(x)
                    output["y"].append(datasets[x][function].mean.get(metric.value))
                    output["err"].append(datasets[x][function].std.get(metric.value))
                ax.errorbar(
                    output["x"],
                    output["y"],
                    yerr=output["err"],
                    label=(f"{function.verbose} - {executor} executors"),
                )
            ax.set_title("{} - {}".format(function.verbose, structure_type.verbose))
            ax.set_ylabel(metric.verbose)
            ax.set_xlabel("Dataset size (unique IDs)")
            ax.set_xscale("log")
            ax.legend()
            figs.append(
                (
                    fig,
                    f"output/{metric.value}/all-data-{structure_type.value}" f"-{function.value}-{metric.value}",
                )
            )
            fig.tight_layout()
        if not os.path.exists(f"output/{metric.value}"):
            os.makedirs(f"output/{metric.value}")
        for fig, path in figs:
            fig.savefig(path)
            fig.close()


def plot_speedup(
    data: ExecutorConfigurationContent,
    function: Function,
    structures: List[StructureType] = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
        # StructureType.AscendingBucketed20,
        # StructureType.AscendingBucketed40,
        # StructureType.AscendingBucketed80,
        # StructureType.AscendingBucketed160,
    ],
    metrics=[
        Metric.ElapsedTime,
    ],
):
    num_executors = sorted(list(data.keys()))
    base_line = num_executors.pop(0)
    for metric in metrics:
        for structure_type in structures:
            fig, ax = plt.subplots()
            for executor in num_executors:
                output: PlotData = {"x": [], "y": [], "err": []}
                datasets = data[executor][structure_type]
                for x in sorted(datasets):
                    output["x"].append(x)
                    output["y"].append(
                        data[base_line][structure_type][x][function].mean.get(metric.value, 0.0)
                        / datasets[x][function].mean.get(metric.value, 0.0)
                    )
                ax.errorbar(
                    output["x"],
                    output["y"],
                    label=(f"{function.verbose} Speedup - {executor} executors"),
                )
            ax.set_title("{} - {}".format(function.verbose, structure_type.verbose))
            ax.set_ylabel("Speedup")
            ax.set_xlabel("Dataset size (unique IDs)")
            ax.set_xscale("log")
            ax.legend()
            plt.tight_layout()
            if not os.path.exists(f"output/{metric.value}"):
                os.makedirs(f"output/{metric.value}")
            plt.savefig(f"output/{metric.value}/speedup-{structure_type.value}" f"-{function.value}-{metric.value}")
            plt.close(fig)


def plot_comparision_executors(
    data: ExecutorConfigurationContent,
    datasize: int,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
    structures: List[StructureType] = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
        StructureType.AscendingBucketed160,
        StructureType.AscendingBucketed80,
        StructureType.AscendingBucketed40,
        StructureType.AscendingBucketed20,
    ],
):
    for metric in Metric:
        for structure_type in sorted(structures):
            output: Dict[Function, PlotData] = {}
            for function in functions:
                output[function] = {"x": [], "y": [], "err": []}

            fig, ax = plt.subplots()

            for num_executor in sorted(data):
                dataset = data[num_executor][structure_type][datasize]
                for function in functions:
                    output[function]["x"].append(num_executor)
                    output[function]["y"].append(dataset[function].mean.get(metric.value))
                    output[function]["err"].append(dataset[function].std.get(metric.value))

            for function in functions:
                ax.errorbar(
                    output[function]["x"],
                    output[function]["y"],
                    output[function]["err"],
                    label=function.verbose,
                )

            ax.set_title(f"{datasize:,} unique IDs - {structure_type.verbose} dataset")
            ax.set_ylabel(metric.verbose)
            ax.set_xlabel("No. executors")
            ax.legend()
            plt.tight_layout()
            if not os.path.exists(f"output/{metric.value}"):
                os.makedirs(f"output/{metric.value}")
            plt.savefig(
                (
                    f"output/{metric.value}/{datasize}-rows-"
                    f"{structure_type.value}-executor_comparision-{metric.value}"
                )
            )
            plt.close(fig)


def plot_comparision_structures(
    data: ExecutorConfigurationContent,
    num_executors: int,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
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
    for metric in Metric:
        figs: List[Tuple[Figure, Axes, str]] = []
        for structure in structures:
            datasets = data[num_executors][structure]
            fig, ax = plt.subplots()
            for function in functions:
                values: PlotData = {"x": [], "y": [], "err": []}
                for x in sorted(datasets):
                    values["x"].append(x)
                    values["y"].append(datasets[x][function].mean.get(metric.value))
                    values["err"].append(datasets[x][function].std.get(metric.value))
                ax.errorbar(values["x"], values["y"], values["err"], label=function.verbose)
            if metric in [
                Metric.ElapsedTime,
                Metric.PeakExecutionMemory,
                Metric.ShuffleBytesWritten,
            ]:
                ax.set_yscale("log")
            ax.set_title(f"{num_executors} executors - {structure.verbose}")
            ax.set_ylabel(metric.verbose)
            ax.set_xlabel("Dataset size (unique IDs)")
            ax.set_xscale("log")
            ax.legend()
            fig.tight_layout()
            figs.append(
                (
                    fig,
                    ax,
                    f"output/{metric.value}/{num_executors}-executors-" f"{structure.value}-{metric.value}",
                )
            )
        if not os.path.exists(f"output/{metric.value}"):
            os.makedirs(f"output/{metric.value}")

        y_min, y_max = min([ax.get_ylim()[0] for _, ax, _ in figs]), max([ax.get_ylim()[1] for _, ax, _ in figs])

        for fig, ax, path in figs:
            ax.set_ylim(y_min, y_max)
            fig.savefig(path)
            plt.close(fig)


def plot_comparision(
    data: ExecutorConfigurationContent,
    num_executors: int,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
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
    for metric in Metric:
        figs: List[Tuple[Figure, Axes, str]] = []
        for function in functions:
            executors = data[num_executors]
            fig, ax = plt.subplots()
            # Combine lines with same values
            combined_structures = _combine_structures(executors, structures, function, metric)
            for structure_types, datasets in combined_structures:
                values: PlotData = {"x": [], "y": [], "err": []}
                for x in sorted(datasets):
                    values["x"].append(x)
                    values["y"].append(datasets[x][function].mean.get(metric.value))
                    values["err"].append(datasets[x][function].std.get(metric.value))
                ax.errorbar(
                    values["x"],
                    values["y"],
                    values["err"],
                    label=" & ".join(structure_type.verbose for structure_type in structure_types),
                )
            if metric in [
                Metric.ElapsedTime,
                Metric.PeakExecutionMemory,
                Metric.ShuffleBytesWritten,
            ]:
                ax.set_yscale("log")
            ax.set_title(f"{num_executors} executors - {function.verbose}")
            ax.set_ylabel(metric.verbose)
            ax.set_xlabel("Dataset size (unique IDs)")
            ax.set_xscale("log")
            ax.legend()
            fig.tight_layout()
            figs.append(
                (
                    fig,
                    ax,
                    f"output/{metric.value}/{num_executors}-executors-" f"{function.value}-{metric.value}",
                )
            )
        if not os.path.exists(f"output/{metric.value}"):
            os.makedirs(f"output/{metric.value}")

        y_min, y_max = min([ax.get_ylim()[0] for _, ax, _ in figs]), max([ax.get_ylim()[1] for _, ax, _ in figs])

        for fig, ax, path in figs:
            ax.set_ylim(y_min, y_max)
            fig.savefig(path)
            plt.close(fig)


def plot_peak_memory(data: ExecutorConfigurationContent):
    metric = Metric.PeakExecutionMemory
    structures = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
    ]
    fig, ax = plt.subplots()
    for function in Function:
        values: PlotData = {"x": [], "y": [], "err": []}
        function_data = _combine_structures(data[2], structures, function, metric)
        assert len(function_data) == 1
        dataset = function_data[0][1]
        for x in sorted(dataset):
            values["x"].append(x)
            values["y"].append(dataset[x][function].mean.get(metric.value))
            values["err"].append(dataset[x][function].std.get(metric.value))
        ax.errorbar(values["x"], values["y"], values["err"], label=function.verbose)
    ax.set_yscale("log")
    ax.set_title("2 executors - Peak memory execution")
    ax.set_ylabel(metric.verbose)
    ax.set_xlabel("Dataset size (unique IDs)")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    if not os.path.exists(f"output/{metric.value}"):
        os.makedirs(f"output/{metric.value}")
    fig.savefig(
        f"output/{metric.value}/2-executors-all-{metric.value}",
    )
    plt.close(fig)


def plot_shuffle_writes(data: ExecutorConfigurationContent):
    metric = Metric.ShuffleBytesWritten
    structures = [
        StructureType.Ascending,
        StructureType.Descending,
        StructureType.Random,
    ]
    fig, ax = plt.subplots()
    for function in Function:
        function_data = _combine_structures(data[2], structures, function, metric)
        assert len(function_data) == 2
        for structure_types, dataset in function_data:
            values: PlotData = {"x": [], "y": [], "err": []}
            for x in sorted(dataset):
                values["x"].append(x)
                values["y"].append(dataset[x][function].mean.get(metric.value))
                values["err"].append(dataset[x][function].std.get(metric.value))
            ax.errorbar(
                values["x"],
                values["y"],
                values["err"],
                label=(f"{function.verbose} - {' & '.join(s.verbose for s in structure_types).replace('order ', '')}"),
                ls=LS_MAP[function],
            )

    ax.set_yscale("log")
    ax.set_title("2 executors - Shuffle bytes written")
    ax.set_ylabel(metric.verbose)
    ax.set_xlabel("Dataset size (unique IDs)")
    ax.set_xscale("log")
    ax.legend(prop={"size": 8})
    fig.tight_layout()
    if not os.path.exists(f"output/{metric.value}"):
        os.makedirs(f"output/{metric.value}")
    fig.savefig(
        f"output/{metric.value}/2-executors-all-{metric.value}",
    )


def plot_buckets_compare(
    data: ExecutorConfigurationContent,
    num_executors: int = 2,
    dataset_size: int = 10_000_000,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
):
    reference_structure = StructureType.Ascending
    bucket_structures: List[Tuple[StructureType, int]] = [
        (StructureType.AscendingBucketed20, 20),
        (StructureType.AscendingBucketed40, 40),
        (StructureType.AscendingBucketed80, 80),
        (StructureType.AscendingBucketed160, 160),
    ]

    for metric in Metric:
        output: Dict[Function, PlotData] = {fun: {"x": [], "y": [], "err": []} for fun in functions}
        refs: Dict[Function, int] = {}

        fig, ax = plt.subplots(figsize=(6.4, 5.8))

        if metric == metric.MemoryBytesSpilled or metric == metric.DiskBytesSpilled:
            ax.ticklabel_format(axis="y", scilimits=(9, 9))

        for function, values in data[num_executors][reference_structure][dataset_size].items():
            refs[function] = values.mean.get(metric.value)

        for bucket_structure, buckets in bucket_structures:
            dataset = data[num_executors][bucket_structure][dataset_size]
            for function in functions:
                output[function]["x"].append(buckets)
                output[function]["y"].append(dataset[function].mean.get(metric.value))
                output[function]["err"].append(dataset[function].std.get(metric.value))

        for function in functions:
            error_bar = ax.errorbar(
                output[function]["x"],
                output[function]["y"],
                output[function]["err"],
                label=function.verbose,
            )
            ax.axhline(
                y=refs[function],
                c=error_bar.lines[0]._color,  # type: ignore
                label=f"{function.verbose} - No buckets, 200 partitions",
                linestyle=(0, (1, 5)),
            )

        ax.set_title(f"{num_executors} executors - {dataset_size:,} unique IDs")
        ax.set_ylabel(metric.verbose)
        ax.set_xlabel("Buckets")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
        plt.tight_layout()
        functions_string = "_".join([function.value for function in functions])
        if not os.path.exists(f"output/{metric.value}"):
            os.makedirs(f"output/{metric.value}")
        plt.savefig(
            (
                "output/{}/buckets-comparision-{}-{}-executors-{}-{}".format(
                    metric.value,
                    dataset_size,
                    num_executors,
                    metric.value,
                    functions_string,
                )
            ),
            bbox_inches="tight",
        )
        plt.close(fig)


def hive_compare(
    data: ExecutorConfigurationContent,
    hive_data: HiveContent,
    functions: List[Function] = [
        Function.Exploding,
        Function.Union,
        Function.EarlyStopSortMerge,
    ],
):
    # Only compare elapsed time for 3 nodes / 8 executors, sorted ascending
    metric = Metric.ElapsedTime
    num_executors = 8
    nodes = 3
    structure_type = StructureType.Ascending
    spark = data[num_executors][structure_type]
    hive = hive_data[nodes][structure_type]

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for function in functions:
        plot_data: PlotData = {"x": [], "y": [], "err": []}
        for size in sorted(spark.keys()):
            plot_data["x"].append(size)
            plot_data["y"].append(spark[size][function].mean.get(metric.value))
            plot_data["err"].append(spark[size][function].std.get(metric.value))

        ax.errorbar(
            plot_data["x"],
            plot_data["y"],
            plot_data["err"],
            label=function.verbose,
        )

    plot_data: PlotData = {"x": [], "y": [], "err": []}
    for size in sorted(hive.keys()):
        plot_data["x"].append(size)
        plot_data["y"].append(hive[size].mean)
        plot_data["err"].append(hive[size].std)

    ax.errorbar(
        plot_data["x"],
        plot_data["y"],
        plot_data["err"],
        label="Tez (Exploding)",
    )
    if metric in [
        Metric.ElapsedTime,
        Metric.PeakExecutionMemory,
        Metric.ShuffleBytesWritten,
    ]:
        ax.set_yscale("log")
    ax.set_title(f"{nodes} nodes, {num_executors} executors, {structure_type.verbose} - Hive comparision")
    ax.set_ylabel(metric.verbose)
    ax.set_xlabel("Dataset size (unique IDs)")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    functions_string = "_".join([function.value for function in functions])
    if not os.path.exists(f"output/{metric.value}"):
        os.makedirs(f"output/{metric.value}")
    plt.savefig(
        (
            "output/{}/hive-comparision-{}-nodes-{}-executors-{}-{}".format(
                metric.value,
                nodes,
                num_executors,
                metric.value,
                functions_string,
            )
        ),
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

    data, hive_data = load_data(args.input)

    # plot_peak_memory(data)
    plot_shuffle_writes(data)
    # plot_comparision_structures(data, 2)
    # for i in [100_000, 1_000_000, 10_000_000]:
    #     plot_buckets_compare(data, 2, i)
    # plot_comparision(data, 2)
    # plot_comparision(data, 8)
    # plot_executors(data, Function.EarlyStopSortMerge)
    # plot_executors(data, Function.Union)
    # plot_executors(data, Function.Exploding)
    # plot_speedup(data, Function.EarlyStopSortMerge)
    # plot_speedup(data, Function.Union)
    # plot_speedup(data, Function.Exploding)

    # plot_comparision_executors(data, 10_000_000)
    # hive_compare(data, hive_data)
