"""
    @file:              tools.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:      This file is used to define some functions used for recording.
"""

from collections import Counter
import json
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
from numpy import max, mean, median, min, std

from src.recording.constants import *
from src.visualization.tools import visualize_importance, visualize_scaled_importance


def get_evaluation_recap(
        evaluation_name: str,
        recordings_path: str
) -> None:
    """
    Creates a file with a summary of results from records.json file of each data split.

    Parameters
    ----------
    evaluation_name : str
        Name of the evaluation.
    recordings_path : str
        Directory where containing the folders with the results of each split.
    """

    # We check if the directory with results exists
    path = os.path.join(recordings_path, evaluation_name)
    if not os.path.exists(path):
        raise ValueError('Impossible to find the given directory')

    # We sort the folders in the directory according to the split number
    folders = next(os.walk(path))[1]
    folders.sort(key=lambda x: int(x.split("_")[1]))

    # Initialization of an empty dictionary to store the summary
    data = {
        TRAIN_METRICS: {},
        TEST_METRICS: {},
        HYPERPARAMETER_IMPORTANCE: {},
        FEATURE_IMPORTANCE: {},
        HYPERPARAMETERS: {},
        COEFFICIENT: {}
    }

    # Initialization of a list of key list that we can found within section of records dictionary
    key_lists = {}

    for folder in folders:

        # We open the json file containing the info of each split
        with open(os.path.join(path, folder, RECORDS_FILE), "r") as read_file:
            split_data = json.load(read_file)

        # For each section and their respective key list
        for section in data.keys():
            if section in split_data.keys():

                # If the key list is not initialized yet..
                if key_lists.get(section) is None:

                    # Initialization of the key list
                    key_lists[section] = split_data[section].keys()

                    # Initialization of each individual key section in the dictionary
                    for key in key_lists[section]:
                        data[section][key] = {VALUES: [], INFO: ""}

                # We add values to each key associated to the current section
                for key in key_lists[section]:
                    data[section][key][VALUES].append(split_data[section][key])

    # We remove empty sections
    data = {k: v for k, v in data.items() if len(v) != 0}

    # We add the info about the mean, the standard deviation, the median , the min, and the max
    set_info(data)

    # We save the json containing the summary of the records
    with open(os.path.join(path, SUMMARY_FILE), "w") as file:
        json.dump(data, file, indent=True)


def set_info(
        data: Dict[str, Dict[str, Union[List[Union[str, float]], str]]]
) -> None:
    """
    Adds the mean, the standard deviation, the median, the min and the max to the numerical parameters of each section
    of the dictionary with the summary. Otherwise, counts the number of appearances of the categorical parameters.

    Parameters
    ----------
    data : Dict[str, Dict[str, Union[List[Union[str, float]], str]]]
        Dictionary with the summary of results from the splits' records.
    """
    # For each section
    for section in data.keys():

        # For each key of this section
        for key in data[section].keys():

            # We extract the list of values
            values = data[section][key][VALUES]

            if not (isinstance(values[0], str) or values[0] is None):
                mean_, std_ = round(mean(values), 4), round(std(values), 4)
                med_, min_, max_ = round(median(values), 4), round(min(values), 4), round(max(values), 4)
                data[section][key][INFO] = f"{mean_} +- {std_} [{med_}; {min_}-{max_}]"
                data[section][key][MEAN] = mean_
                data[section][key][STD] = std_
            else:
                counts = Counter(data[section][key][VALUES])
                data[section][key][INFO] = str(dict(counts))


def plot_hps_importance_chart(
        evaluation_name: str,
        recordings_path: str
) -> None:
    """
    Creates a bar plot containing information about the mean and standard deviation of each hyperparameter's importance.

    Parameters
    ----------
    evaluation_name : str
        Name of the evaluation.
    recordings_path : str
        Directory where containing the folders with the results of each split.
    """
    # We get the content of the json file
    path = os.path.join(recordings_path, evaluation_name)
    with open(os.path.join(path, SUMMARY_FILE), "r") as read_file:
        data = json.load(read_file)[HYPERPARAMETER_IMPORTANCE]

    # We create the bar plot
    visualize_importance(
        data=data,
        figure_title='HPs importance',
        filename=os.path.join(path, HPS_IMPORTANCE_CHART)
    )


def plot_feature_importance_charts(
        evaluation_name: str,
        recordings_path: str
) -> None:
    """
    Creates a bar plots containing information about the mean and standard deviation of each feature's importance.

    Parameters
    ----------
    evaluation_name : str
        Name of the evaluation.
    recordings_path : str
        Directory where containing the folders with the results of each split.
    """
    # We get the content of the json file
    path = os.path.join(recordings_path, evaluation_name)
    with open(os.path.join(path, SUMMARY_FILE), "r") as read_file:
        data = json.load(read_file)[FEATURE_IMPORTANCE]

    # We create the bar plots
    visualize_importance(
        data=data,
        figure_title='Features importance',
        filename=os.path.join(path, FEATURE_IMPORTANCE_CHART)
    )
    visualize_scaled_importance(
        data=data,
        figure_title='Scaled feature importance',
        filename=os.path.join(path, S_FEATURE_IMPORTANCE_CHART)
    )


def compare_prediction_recordings(
        evaluations: List[str],
        split_index: int,
        recording_path: str
) -> None:
    """
    Creates a scatter plot showing the predictions of one or two experiments against the real labels.

    Parameters
    ----------
    evaluations : List[str]
        List of str representing the names of the evaluations to compare.
    split_index : int
        Index of the split we want to compare.
    recording_path : str
        Directory that stores the evaluations folder.
    """

    # We check that the number of evaluations provided is 2
    if not (1 <= len(evaluations) <= 2):
        raise ValueError("One or two evaluations must be specified")

    # We create the paths to recoding files
    paths = [os.path.join(recording_path, e, f"Split_{split_index}", RECORDS_FILE) for e in evaluations]

    # We get the data from the recordings
    all_data = []  # List of dictionaries
    for path in paths:

        # We read the record file of the first evaluation
        with open(path, "r") as read_file:
            all_data.append(json.load(read_file))

    # We check if the two evaluations are made on the same patients
    comparison_possible = True
    first_experiment_ids = list(all_data[0][TEST_RESULTS].keys())

    for i, data in enumerate(all_data[1:]):

        # We check the length of both predictions list
        if len(data[TEST_RESULTS]) != len(all_data[0][TEST_RESULTS]):
            comparison_possible = False
            break

        # We check ids in both list
        for j, id_ in enumerate(data[TEST_RESULTS].keys()):
            if id_ != first_experiment_ids[j]:
                comparison_possible = False
                break

    if not comparison_possible:
        raise ValueError("Different patients are present in the given evaluations")

    targets, ids, all_predictions = [], [], []

    # We gather the needed data from the recordings
    for i, data in enumerate(all_data):

        # We add an empty list to store predictions
        all_predictions.append([])

        for id_, item in data[TEST_RESULTS].items():

            # If we have not registered ids and targets yet
            if i == 0:
                ids.append(id_)
                targets.append(float(item[TARGET]))

            all_predictions[i].append(float(item[PREDICTION]))

    # We sort predictions and the ids based on their targets
    indexes = list(range(len(targets)))
    indexes.sort(key=lambda x: targets[x])
    all_predictions = [[predictions[i] for i in indexes] for predictions in all_predictions]
    targets = [targets[i] for i in indexes]
    ids = [ids[i] for i in indexes]

    # We set some parameters of the plot
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.rcParams["xtick.labelsize"] = 6

    # We create the scatter plot
    colors = ["blue", "orange"]
    plt.scatter(ids, targets, color="green", label="ground truth")
    for i, predictions in enumerate(all_predictions):
        plt.scatter(ids, predictions, color=colors[i], label=evaluations[i])

    # We add the legend and the title to the plot
    plt.legend()
    plt.title("Predictions and ground truth")

    # We save the plot
    plt.savefig(
        os.path.join(recording_path, evaluations[0], f"Split_{split_index}", f"comparison_{'_'.join(evaluations)}.png"))
    plt.close()
