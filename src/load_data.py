import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Match, cast

import pandas as pd

EXECUTOR_DIR_REGEX = r"^(\d+)\-executors$"
STRUCTURE_TYPE_REGEX = r"^([a-zA-Z\-]+)$"
DATASET_REGEX = r"^(\d+)\-1\_year$"
CSV_REGEX = r"^[a-zA-Z\-0-9]+.csv$"


class Function(Enum):
    Exploding = "exploding"
    Union = "union"
    EarlyStopSortMerge = "earlyStopSortMerge"

    @property
    def verbose(self):
        return FUNCTION_VERBOSE_NAME[self]


class StructureType(Enum):
    Ascending = "sorted-asc"
    Descending = "sorted-desc"
    Random = "sorted-rand"

    @property
    def verbose(self):
        return STRUCTURE_TYPE_VERBOSE_NAME[self]


FUNCTION_VERBOSE_NAME: Dict[Function, str] = {
    Function.Exploding: "Exploding",
    Function.Union: "Union",
    Function.EarlyStopSortMerge: "Early Stop Sort-Merge",
}

STRUCTURE_TYPE_VERBOSE_NAME: Dict[StructureType, str] = {
    StructureType.Ascending: "Ascending order",
    StructureType.Descending: "Decending order",
    StructureType.Random: "Random order",
}


@dataclass
class FunctionInfo:
    key: Function
    path: str


FUNCTIONS: List[FunctionInfo] = [
    FunctionInfo(Function.Exploding, "exploding_experiment.csv"),
    FunctionInfo(Function.Union, "union_experiment.csv"),
    FunctionInfo(Function.EarlyStopSortMerge, "early_stop_sort_merge.csv"),
]


@dataclass
class DataFrameTypes:
    raw: pd.DataFrame
    median: pd.Series
    mean: pd.Series
    std: pd.Series


FunctionOutput = Dict[Function, DataFrameTypes]


def get_function_output(directory: str) -> FunctionOutput:
    outputs: FunctionOutput = {}  # type: ignore
    for function in FUNCTIONS:
        path = f"{directory}/{function.path}"
        output_file = [
            cast(Match[str], re.match(CSV_REGEX, function_output)).group(0)
            for function_output in os.listdir(path)
            if re.match(CSV_REGEX, function_output)
        ][0]
        full_file_path = f"{path}/{output_file}"
        dataframe = pd.read_csv(full_file_path)
        outputs[function.key] = DataFrameTypes(
            raw=dataframe,
            median=dataframe.median(),
            mean=dataframe.mean(),
            std=dataframe.std(),
        )
    return outputs


DatasetContent = Dict[int, FunctionOutput]


def get_datasets(directory: str) -> DatasetContent:
    datasets = [
        int(cast(Match[str], re.match(DATASET_REGEX, structure_type)).group(1))
        for structure_type in os.listdir(directory)
        if re.match(DATASET_REGEX, structure_type)
    ]
    dataset_content: DatasetContent = {}
    for dataset in datasets:
        dataset_content[dataset] = get_function_output(f"{directory}/{dataset}-1_year")
    return dataset_content


StructureTypeContent = Dict[StructureType, DatasetContent]


def get_executor_structure_types(directory: str) -> StructureTypeContent:
    structure_types = [
        cast(Match[str], re.match(STRUCTURE_TYPE_REGEX, structure_type)).group(1)
        for structure_type in os.listdir(directory)
        if re.match(STRUCTURE_TYPE_REGEX, structure_type)
    ]

    structure_type_content: StructureTypeContent = {}
    for structure_type_str in structure_types:
        # Skip the raw
        if structure_type_str == "raw":
            continue

        structure_type = StructureType(structure_type_str)
        structure_type_content[structure_type] = get_datasets(
            f"{directory}/{structure_type.value}"
        )
    return structure_type_content


ExecutorConfigurationContent = Dict[int, StructureTypeContent]


def load_data(directory: str) -> ExecutorConfigurationContent:
    executor_configurations = [
        int(cast(Match[str], re.match(EXECUTOR_DIR_REGEX, executor_dir)).group(1))
        for executor_dir in os.listdir(directory)
        if re.match(EXECUTOR_DIR_REGEX, executor_dir)
    ]
    executor_configuration_content: ExecutorConfigurationContent = {}
    for executor_configuration in executor_configurations:
        executor_configuration_content[
            executor_configuration
        ] = get_executor_structure_types(
            f"{directory}/{executor_configuration}-executors"
        )
    return executor_configuration_content
