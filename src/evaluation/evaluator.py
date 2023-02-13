"""
    @file:              evaluation.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     03/2022
    @Last modification: 02/2023

    @Description:      This file is used to define the Evaluator class in charge of comparing models against each other.
"""

from copy import deepcopy
from json import load
from os import makedirs, path
from time import strftime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.random import seed as np_seed
from pandas import DataFrame
import ray
from torch import is_tensor, from_numpy, manual_seed, stack

from ..data.datasets import ProstateCancerDataset, TableDataset
from ..data.processing.feature_selection import FeatureSelector
from ..data.processing.sampling import Mask
from ..models.base.base_model import BaseModel
from ..recording.constants import (PREDICTION, RECORDS_FILE, TEST_RESULTS, TRAIN_RESULTS, VALID_RESULTS)
from ..recording.recorder import Recorder
from ..recording.tools import (compare_prediction_recordings, get_evaluation_recap, plot_feature_importance_charts,
                                 plot_hps_importance_chart)
from ..tasks import BinaryClassificationTask, SegmentationTask
from ..tuning import Objective, Tuner


class Evaluator:
    """
    Object in charge of evaluating a model over multiple different data splits.
    """

    def __init__(
            self,
            model_constructor: Callable,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
            hps: Dict[str, Dict[str, Any]],
            n_trials: int,
            path_to_experiment_records: str,
            fixed_params: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            gpu_device: bool = False,
            evaluation_name: Optional[str] = None,
            feature_selector: Optional[FeatureSelector] = None,
            fixed_params_update_function: Optional[Callable] = None,
            save_hps_importance: Optional[bool] = False,
            save_parallel_coordinates: Optional[bool] = False,
            save_pareto_front: Optional[bool] = False,
            save_optimization_history: Optional[bool] = False,
            pred_path: Optional[str] = None
    ):
        """
        Set protected and public attributes.

        Parameters
        ----------
        model_constructor : Callable
            Callable object used to generate a model according to a set of hyperparameters.
        dataset : ProstateCancerDataset
            Custom dataset containing the whole learning dataset needed for our evaluations.
        masks : Dict[int, Dict[str, List[int]]]
            Dict with list of idx to use as train, valid and test masks.
        hps : Dict[str, Dict[str, Any]]
            Dictionary with information on the hyperparameters we want to tune.
        fixed_params : Optional[Dict[str, Any]]
            Dictionary with parameters used by the model constructor for building model
        n_trials : int
            Number of hyperparameters sets sampled within each inner validation loop.
        path_to_experiment_records : str
            Path to experiment records (A directory).
        seed : Optional[int]
            Random state used for reproducibility.
        gpu_device : bool
            True if we want to use the gpu.
        evaluation_name : Optional[str]
            Name of the results file saved at the recordings_path.
        feature_selector : Optional[FeatureSelector]
            Object used to proceed to feature selection during nested cross valid.
        fixed_params_update_function : Optional[Callable]
            Function that updates fixed params dictionary from ProstateCancerSubset after feature selection. Might be
            necessary for model with entity embedding.
        save_hps_importance : Optional[bool]
            True if we want to plot the hyperparameters importance graph after tuning.
        save_parallel_coordinates : Optional[bool]
            True if we want to plot the parallel coordinates graph after tuning.
        save_pareto_front : Optional[bool]
            Whether we want to plot the pareto front after tuning.
        save_optimization_history : Optional[bool]
            True if we want to plot the optimization history graph after tuning.
        pred_path : Optional[str]
            If given, the path will be used to load predictions from another experiment and include them within actual
            features.
        """

        # We look if a file with the same evaluation name exists
        if evaluation_name is not None:
            if path.exists(path.join(path_to_experiment_records, evaluation_name)):
                raise ValueError("evaluation with this name already exists")
        else:
            makedirs(path_to_experiment_records, exist_ok=True)
            evaluation_name = f"{strftime('%Y%m%d-%H%M%S')}"

        # We set protected attributes
        self._dataset = dataset
        self._gpu_device = gpu_device
        self._feature_selector = feature_selector
        self._feature_selection_count = {feature: 0 for feature in self._dataset.table_dataset.original_data.columns}
        self._fixed_params = fixed_params if fixed_params is not None else {}
        self._hps = hps
        self._masks = masks
        self._path_to_experiment_records = path_to_experiment_records
        self._pred_path = pred_path
        self._hp_tuning = (n_trials > 0)
        self._tuner = Tuner(
            n_trials=n_trials,
            save_hps_importance=save_hps_importance,
            save_parallel_coordinates=save_parallel_coordinates,
            save_pareto_front=save_pareto_front,
            save_optimization_history=save_optimization_history,
            path=path_to_experiment_records
        )

        # We set the public attributes
        self.evaluation_name = evaluation_name
        self.model_constructor = model_constructor
        self.seed = seed

        # We set the fixed params update method
        if fixed_params_update_function is not None:
            self._update_fixed_params = fixed_params_update_function
        else:
            self._update_fixed_params = lambda _: self._fixed_params

    def _extract_table_subset(
            self,
            records_path: str,
            recorder: Recorder
    ) -> TableDataset:
        """
        Executes the feature selection process and save a record of the procedure at the "records_path".

        Parameters
        ----------
        records_path : str
            Directory where the feature selection record will be save.
        recorder : Recorder
            A recorder.

        Returns
        -------
        subset : TableDataset
            Columns subset of the current dataset
        """
        # Creation of subset using feature selection
        if self._feature_selector is not None:
            cont_cols, cat_cols, fi_dict = self._feature_selector(
                dataset=self._dataset.table_dataset,
                records_path=records_path,
                return_imp=True
            )
            recorder.record_features_importance(fi_dict)
            subset = self._dataset.table_dataset.create_subset(cont_cols=cont_cols, cat_cols=cat_cols)
        else:
            subset = deepcopy(self._dataset.table_dataset)

        # Update of feature appearances count
        for feature in subset.original_data.columns:
            self._feature_selection_count[feature] += 1

        return subset

    def evaluate(self) -> None:
        """
        Performs nested subsampling validations to evaluate a model and saves results in specific files using a
        recorder.
        """
        # We set the seed for the nested subsampling valid procedure
        if self.seed is not None:
            np_seed(self.seed)
            manual_seed(self.seed)

        # We initialize ray
        ray.init()

        # We execute the outer loop
        for k, v in self._masks.items():

            # We extract the masks
            train_mask, valid_mask = v[Mask.TRAIN], v[Mask.VALID]
            test_mask, in_masks = v[Mask.TEST], v[Mask.INNER]

            # We update the dataset's masks
            self._dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

            # We create the Recorder object to save the result of this experience
            recorder = Recorder(
                evaluation_name=self.evaluation_name,
                index=k,
                recordings_path=self._path_to_experiment_records
            )

            # We save the saving path
            saving_path = path.join(self._path_to_experiment_records, self.evaluation_name, f"Split_{k}")

            # We proceed to feature selection
            table_subset = self._extract_table_subset(records_path=saving_path, recorder=recorder)

            # We create a ProstateCancerDataSubset
            subset = ProstateCancerDataset(
                image_dataset=self._dataset.image_dataset,
                table_dataset=table_subset
            )

            # We include predictions from another experiment if needed
            if self._pred_path is not None:
                table_subset = self._load_predictions(split_number=k, subset=table_subset)

            # We update the fixed parameters according to the subset
            self._fixed_params = self._update_fixed_params(table_subset)

            # We record the data count
            for name, mask in [("train_set", train_mask), ("valid_set", valid_mask), ("test_set", test_mask)]:
                mask_length = len(mask) if mask is not None else 0
                recorder.record_data_info(name, mask_length)

            # We update the tuner to perform the hyperparameters optimization
            if self._hp_tuning:

                print(f"\nHyperparameter tuning started - K = {k}\n")

                # We update the tuner
                self._tuner.update_tuner(
                    study_name=f"{self.evaluation_name}_{k}",
                    objective=self._create_objective(masks=in_masks, subset=subset),
                    saving_path=saving_path
                )

                # We perform the hps tuning to get the best hps
                best_hps, hps_importance = self._tuner.tune()

                # We save the hyperparameters
                print(f"\nHyperparameter tuning done - K = {k}\n")
                recorder.record_hyperparameters(best_hps)

                # We save the hyperparameters importance
                recorder.record_hyperparameters_importance(hps_importance)
            else:
                best_hps = {}

            # We create a model with the best hps
            model = self.model_constructor(**best_hps, **self._fixed_params)

            # We train our model with the best hps
            print(f"\nFinal model training - K = {k}\n")
            subset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            model.fit(dataset=subset)

            # We save plots associated to training
            if hasattr(model, 'plot_evaluations'):
                model.plot_evaluations(save_path=saving_path)

            # We save the trained model
            model.save_model(path=saving_path)

            # We get the predictions and save the evaluation metric scores
            self._record_scores_and_pred(model, recorder, subset)

            # We save all the data collected in one file
            recorder.generate_file()

            # We generate a plot that compares predictions to ground_truth
            # compare_prediction_recordings(
            #     evaluations=[self.evaluation_name],
            #     split_index=k,
            #     recording_path=self._path_to_experiment_records
            # )

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=self.evaluation_name, recordings_path=self._path_to_experiment_records)

        # We save the features importance charts
        if self._feature_selector is not None:
            plot_feature_importance_charts(
                evaluation_name=self.evaluation_name,
                recordings_path=self._path_to_experiment_records
            )

        # We save the hyperparameters importance chart
        if self._hp_tuning:
            plot_hps_importance_chart(
                evaluation_name=self.evaluation_name,
                recordings_path=self._path_to_experiment_records
            )

        # We shutdown ray
        ray.shutdown()

    def _create_objective(
            self,
            masks: Dict[int, Dict[str, List[int]]],
            subset: ProstateCancerDataset
    ) -> Objective:
        """
        Creates an adapted objective function to pass to our tuner.

        Parameters
        ----------
        masks : Dict[int, Dict[str, List[int]]]
            Inner masks for hyperparameters tuning.
        subset : ProstateCancerDataset
            Subset of the original dataset after feature selection.

        Returns
        -------
        objective : Objective
            Objective function.
        """
        return Objective(
            dataset=subset,
            masks=masks,
            hps=self._hps,
            fixed_params=self._fixed_params,
            model_constructor=self.model_constructor,
            gpu_device=self._gpu_device
        )

    def _load_predictions(
            self,
            split_number: int,
            subset: TableDataset
    ) -> TableDataset:
        """
        Loads prediction in a given path and includes them as a feature within the dataset.

        Parameters
        ----------
        split_number : int
            Split for which we got to load predictions.
        subset : ProstateCancerDataset
            Actual dataset.

        Returns
        -------
        updated_dataset : TableDataset
            Update dataset.
        """

        # Loading of records
        with open(path.join(self._pred_path, f"Split_{split_number}", RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        # We check the format of predictions
        random_pred = list(data[TRAIN_RESULTS].values())[0][PREDICTION]
        if "[" not in random_pred:

            # Saving of the number of predictions columns
            nb_pred_col = 1

            # Creation of the conversion function to extract predictions from strings
            def convert(x: str) -> List[float]:
                return [float(x)]
        else:

            # Saving of the number of predictions columns
            nb_pred_col = len(random_pred[1:-1].split(','))

            # Creation of the conversion function to extract predictions from strings
            def convert(x: str) -> List[float]:
                return [float(a) for a in x[1:-1].split(',')]

        # Extraction of predictions
        pred = {}
        for section in TRAIN_RESULTS, TEST_RESULTS, VALID_RESULTS:
            pred = {**pred, **{p_id: [p_id, *convert(v[PREDICTION])] for p_id, v in data[section].items()}}

        # Creation of pandas dataframe
        pred_col_names = [f'pred{i}' for i in range(nb_pred_col)]
        df = DataFrame.from_dict(pred, orient='index', columns=[subset.ids_col, *pred_col_names])

        # Creation of new augmented dataset
        return subset.create_superset(data=df, categorical=False)

    @staticmethod
    def _record_scores_and_pred(
            model: BaseModel,
            recorder: Recorder,
            subset: ProstateCancerDataset
    ) -> None:
        """
        Records the scores associated to train and test set and also saves the prediction linked to each individual.

        Parameters
        ----------
        model : BaseModel
            Model trained with best found hyperparameters.
        recorder : Recorder
            Object recording information about splits evaluations.
        subset : ProstateCancerDataset
            Dataset with remaining features from feature selection.
        """

        # We find the optimal threshold and save it
        model.fix_thresholds_to_optimal_values(dataset=subset)

        # Record thresholds
        classification_tasks = [task for task in subset.tasks if isinstance(task, BinaryClassificationTask)]
        for task in classification_tasks:
            for metric in task.metrics:
                recorder.record_data_info(f"{task.name}_{metric.name}_Threshold", str(metric.threshold))

        for mask, mask_type in [
            (subset.train_mask, Mask.TRAIN),
            (subset.test_mask, Mask.TEST),
            (subset.valid_mask, Mask.VALID)
        ]:

            if len(mask) > 0:

                # We compute predictions
                predictions = model.predict_dataset(dataset=subset, mask=mask)

                # We compute scores
                scores = model.scores_dataset(dataset=subset, mask=mask)

                # We extract ids
                ids = [subset.table_dataset.ids[i] for i in mask]

                # We extract table targets and create tensors
                targets = {task.name: [] for task in subset.tasks if isinstance(task, SegmentationTask)}
                for sample in subset[mask]:
                    for task_name in targets.keys():
                        targets[task_name].append(sample.y[task_name])

                if subset.table_dataset.to_tensor:
                    targets = {k: stack(v, dim=0) for k, v in targets.items()}
                else:
                    targets = {k: np.stack(v, axis=0) for k, v in targets.items()}

                # We record all tasks and metric scores
                for task_name, metrics in scores.items():
                    for metric_name, score in metrics.items():
                        recorder.record_scores(score=score, metric=metric_name, mask_type=mask_type, task=task_name)

                for task in subset.tasks:
                    if isinstance(task, SegmentationTask):
                        continue

                    pred = predictions[task.name]

                    if not is_tensor(pred):
                        pred = from_numpy(pred)

                    if isinstance(task, BinaryClassificationTask):
                        # We get the final predictions from the soft predictions
                        pred = (pred >= task.decision_threshold_metric.threshold).long()

                    # We save the predictions
                    recorder.record_predictions(
                        predictions=pred,
                        ids=ids,
                        targets=targets[task.name],
                        mask_type=mask_type
                    )
