"""
    @file:              cleaning.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains the 'Cleaner' object used to remove invalid rows and columns from learning
                        dataframes. This class also indicates warning about potential outliers in the dataset.
"""

from json import dump
from os import makedirs
from os.path import join
from typing import List, Optional, Tuple

from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .tools import retrieve_numerical_var
from ..transforms import Normalization


class Cleaner:
    """
    Object used to clean up the data frames to ensure that our learning task is done correctly. It removes rows and
    columns with too many missing values (percentage above thresh) and then identifies outliers according to their
    numerical attributes.
    """
    # RECORDS CSTS
    CP = "Critical Patients"
    CC = "Critical Columns"
    CQD = "Column Quartile Details"
    CSC = "Chi-Squared Cutoff"

    # PRE-SAVED WARNING MESSAGES
    ROW_THRESHOLD_WARNING = "Row threshold condition not fulfilled."
    COL_THRESHOLD_WARNING = "Column threshold condition not fulfilled."

    # OUTLIER LVLS
    LOW = "LOW"
    HIGH = "HIGH"

    # BOXES DISTANCE
    BD = 2

    # MAHALANOBIS COLUMN ID
    MAHALANOBIS = "Mahalanobis"

    def __init__(
            self,
            records_path: str,
            ids_col: str,
            column_thresh: float = 0.15,
            row_thresh: float = 0.20,
            outlier_alpha: float = 1.5,
            min_n_per_cat: int = 8,
            max_cat_percentage: float = 0.95,
            qchi2_mahalanobis_cutoff: float = 0.975
    ):
        """
        Saves private attributes.

        Parameters
        ----------
        records_path : str
            json file path to save results of cleaning
        ids_col : str
            Name of the column containing the patient ids.
        column_thresh : float
            Percentage threshold (0 <= thresh <= 1).
        row_thresh : float
            Percentage threshold (0 <= thresh <= 1)
        outlier_alpha : float
            Constant multiplied by interquartile range (IQR) to determine uni-variate outliers.
        min_n_per_cat : int
            Minimal number of items having a certain category value in a categorical column.
        max_cat_percentage : float
            Maximal percentage that a category can occupied within a categorical column.
        qchi2_mahalanobis_cutoff : float
            Chi-squared quantile probability used to determine Mahalanobis cutoff value for multivariate outliers.
        """
        # We validate parameters choices
        if not (0 <= column_thresh <= 1 and 0 <= row_thresh <= 1):
            raise ValueError("Thresholds must be in range [0, 1]")

        if min_n_per_cat < 0:
            raise ValueError("The minimal number of items per category must be greater than 0")

        if not (0 < max_cat_percentage < 1):
            raise ValueError("The maximal percentage must be in range (0, 1)")

        if not (0 < qchi2_mahalanobis_cutoff < 1):
            raise ValueError("Chi-squared quantile cutoff must be in range (0, 1)")

        # Internal private fixed attributes
        self.__ids_col = ids_col
        self.__column_thresh = column_thresh
        self.__row_thresh = row_thresh
        self.__outlier_alpha = outlier_alpha
        self.__min_n_per_cat = min_n_per_cat
        self.__max_cat_percentage = max_cat_percentage
        self.__normalize = Normalization()
        self.__qchi2 = qchi2_mahalanobis_cutoff
        self.__records_path = records_path
        self.__plots_path = join(records_path, "plots")

        # Private mutable attribute
        self.__records = {
            self.CP: {},
            self.CC: {},
            "Column Threshold": self.__column_thresh,
            "Row Threshold": self.__row_thresh,
            "Outlier Alpha": self.__outlier_alpha,
            "Min Item Per Category": self.__min_n_per_cat,
            self.CSC: {"Probability": self.__qchi2},
            self.CQD: {},
            self.MAHALANOBIS: {}
        }

        # Creation of folder to store results
        makedirs(self.__plots_path, exist_ok=True)

    def __call__(
            self,
            df: pd.DataFrame,
            numerical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Executes all the cleaning steps and saves records of the procedure.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.
        numerical_columns : Optional[List[str]]
            List with names of numerical columns.

        Returns
        -------
        cleaned_df : pd.DataFrame
            Cleaned pandas dataframe.
        """
        # We find numerical columns if none are mentioned
        if numerical_columns is None:
            numerical_df = retrieve_numerical_var(df, [])
            numerical_columns = list(numerical_df.columns.values)

        # We set missing values to NaN
        df = df.fillna(np.nan)

        # We identify and remove categories of categorical column with a too low number of appearances.
        # We remove columns with categories that number of appearances is too high
        categorical_columns = [c for c in df.columns.values if c not in [self.__ids_col] + numerical_columns]
        cleaned_df = self.__identify_critical_categories(df, categorical_columns)

        # We identify and remove columns and rows with too many missing values
        cleaned_df = self.__identify_critical_rows_and_columns(cleaned_df)

        # With some rows now removed, we got to check again the validity of columns
        categorical_columns = [c for c in categorical_columns if c in cleaned_df.columns.values]
        cleaned_df = self.__identify_critical_categories(cleaned_df, categorical_columns)

        # We make sure that numerical columns values are float and fill NaN with columns' means
        updated_df = self.__refactor_dataframe(cleaned_df, numerical_columns)

        # Creation of boxplots for each numerical attribute separately
        # Recording of potential univariate outliers and recording of attributes' quartiles
        for c in numerical_columns:
            self.__identify_univariate_outliers(updated_df, [c])

        # Creation of boxplot for all numerical attributes at once
        self.__identify_univariate_outliers(updated_df, numerical_columns, record=False)

        # Recording of potential multivariate outliers and their mahalanobis distances
        self.__identify_multivariate_outliers(updated_df, numerical_columns)

        # We save the records in a .json file
        self.__save_records()

        return cleaned_df

    def __identify_critical_categories(
            self,
            df: pd.DataFrame,
            categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Sets categories with less than "self.__min_n_per_cat" to NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.
        categorical_columns : List[str]
            List with names of categorical columns.

        Returns
        -------
        curated_df : pd.DataFrame
            Curated pandas dataframe.
        """
        # We save a list with column to remove
        to_remove = []

        # For each categorical column
        for c in categorical_columns:

            # We identify the possible categories
            categories = [cat for cat in df[c].unique() if cat is not np.nan]

            # If one category does not satisfy the min_n_per_cat threshold
            # we save a warning in the records and change the value for NaN.
            cat_counts = {}
            for cat in categories:

                cat_counts[cat] = df.loc[df[c] == cat].shape[0]
                if cat_counts[cat] < self.__min_n_per_cat:
                    warning = self.__return_categorical_warning(cat, cat_counts[cat])
                    self.__update_single_column_records(c, warning)
                    df.loc[df[c] == cat, [c]] = np.nan

            # If one category does not satisfy the max_cat_percentage threshold
            # we remove the column (NaNs are included in categories count)
            nb_nan = df.loc[df[c].isnull()].shape[0]
            for cat in categories:

                p = (cat_counts[cat] + nb_nan)/df.shape[0]
                if p > self.__max_cat_percentage:
                    warning = self.__return_categorical_warning(cat, p, min_warning=False)
                    self.__update_single_column_records(c, warning)
                    to_remove.append(c)

        # We delete columns to remove
        df = df.drop(to_remove, axis=1)

        return df

    def __identify_critical_rows_and_columns(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Removes rows and columns with too many missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.

        Returns
        -------
        curated_df : pd.DataFrame
            Curated pandas dataframe.
        """
        # Column cleaning
        updated_df = df.dropna(axis=1, thresh=round(df.shape[0]*(1-self.__column_thresh)))

        # Row cleaning
        updated_df = updated_df.dropna(axis=0, thresh=round(updated_df.shape[1]*(1-self.__row_thresh)))

        # Records update
        for participant in df[self.__ids_col].values:
            if participant not in updated_df[self.__ids_col].values:
                self.__records[self.CP][participant] = {0: self.ROW_THRESHOLD_WARNING}

        for column in df.columns:
            if column not in updated_df.columns:
                self.__update_single_column_records(column, self.COL_THRESHOLD_WARNING)

        return updated_df

    def __identify_univariate_outliers(
            self,
            df: pd.DataFrame,
            numerical_columns: List[str],
            record: bool = True,
            figure_format: str = "png"
    ) -> None:
        """
        Identifies patients (rows) with numerical attribute value lower than Q1 - alpha*(Q3-Q1) or higher than
        Q3 + alpha*(Q3-Q1) and creates a plot.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.
        numerical_columns : Optional[List[str]]
            List with names of numerical columns.
        record : bool
            True indicates that we want to save outliers and quartiles in the records
        figure_format : str
            Figure format.
        """
        # Check if there is a single box or there are multiple boxes to create
        nb_numerical_col = len(numerical_columns)
        single_box = (nb_numerical_col == 1)

        # Creation of boxplot figure
        fig, ax = plt.subplots()
        updated_df = df

        if not single_box:

            # Saving of figure filename
            filename = join(self.__plots_path, "boxplot")

            # Normalization of column values
            updated_df[numerical_columns] = self.__normalize(df[numerical_columns])

            # Creation of figure with multiple boxplots
            data_dict = updated_df[numerical_columns].to_dict('list')
            bp = ax.boxplot(data_dict.values(), whis=self.__outlier_alpha,
                            positions=[i*self.BD for i in range(nb_numerical_col)])

            ax.set_xticklabels(data_dict.keys(), rotation=45)

        else:
            # Saving of figure filename
            filename = join(self.__plots_path, f"{numerical_columns[0].replace('/','-')} boxplot")

            # Creation of single boxplot
            bp = ax.boxplot(df[numerical_columns[0]].values, whis=self.__outlier_alpha, positions=[0])

            # Removal of x label
            plt.tick_params(
                axis='x',           # changes apply to the x-axis
                which='both',       # both major and minor ticks are affected
                bottom=False,       # ticks along the bottom edge are off
                top=False,          # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            ax.set_ylabel(numerical_columns[0])

        ax.set_title(f'Potential univariate outliers (a = {self.__outlier_alpha})')

        # Identification and recording of potential outliers
        texts = []
        for i, c in enumerate(numerical_columns):

            # Quartiles extraction
            q1, q2, q3, iqr = self.__extract_quartiles(bp, i)

            if record:
                self.__records[self.CQD][c] = {"Q1": q1, "Q2": q2, "Q3": q3, "IQR": iqr}

            # Outliers identification (< Q1)
            low_outliers = updated_df.loc[updated_df[c] < q1 - self.__outlier_alpha*iqr, [self.__ids_col, c]]
            self.__update_outliers_records(low_outliers, c, self.LOW, i*self.BD, ax, texts, record)

            # Outliers identification (> Q3)
            high_outliers = updated_df.loc[updated_df[c] > q3 + self.__outlier_alpha*iqr, [self.__ids_col, c]]
            self.__update_outliers_records(high_outliers, c, self.HIGH, i*self.BD, ax, texts, record)

        # We save the box plot
        adjust_text(texts, only_move={'points': 'y', 'texts': 'y'},
                    arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        fig.tight_layout()
        fig.savefig(f"{filename}.{figure_format}", format=figure_format)

    def __identify_multivariate_outliers(
            self,
            df: pd.DataFrame,
            numerical_columns: List[str],
            figure_format: str = "png"
    ) -> None:
        """
        Identifies patients with Mahalanobis distances abnormally high (over the qchi2 quantile of Chi-squared
        distribution with D degrees of freedom) where D is the number of numerical dimensions in our dataframe (after
        cleaning). See the following that explains why squared mahalanobis distances follow a Chi-squared
        distribution : https://markusthill.github.io/mahalanbis-chi-squared/.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.
        numerical_columns : Optional[List[str]]
            List with names of numerical columns.
        figure_format : str
            Figure format.
        """

        # We calculate Mahalanobis distances for all patients
        temporary_df = df.loc[:, [self.__ids_col]]
        temporary_df[self.MAHALANOBIS] = self.__calculate_squared_mahalanobis_distances(df, numerical_columns)

        # We calculate a cutoff values based on a Chi-squared distribution
        cutoff = chi2.ppf(self.__qchi2, len(numerical_columns))
        self.__records[self.CSC].update({"Quantile": round(cutoff, 4)})

        # We order distances and create a plot
        temporary_df.sort_values(by=self.MAHALANOBIS, inplace=True)
        fig, ax = plt.subplots()
        ax.scatter(range(temporary_df.shape[0]), temporary_df[self.MAHALANOBIS])
        ax.hlines(cutoff, xmin=0, xmax=temporary_df.shape[0], linestyles='dashed', colors='black')
        texts = []

        # We save distances to records
        for i in range(temporary_df.shape[0]):

            # We extract patient's id and value
            patient, value = temporary_df.iloc[i, 0], round(temporary_df.iloc[i, 1], 4)
            self.__records[self.MAHALANOBIS][patient] = value

            # We add a warning message to the patient if its value is over the cutoff
            if value > cutoff:
                texts.append(ax.text(i, value, patient))
                warning = self.__return_mahalanobis_warning(value)
                self.__update_single_patient_records(patient, warning)

        # We adjust plot axis
        ax.set_ylabel(f"Squared {self.MAHALANOBIS} distance")
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        # We save the plot
        adjust_text(texts, only_move={'points': 'y', 'texts': 'y'},
                    arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        ax.set_title(f'Potential mutlivariate outliers (qchi2 = {self.__qchi2})')
        fig.savefig(f"{join(self.__plots_path, self.MAHALANOBIS)}.{figure_format}", format=figure_format)

    def __update_outliers_records(
            self,
            subset: pd.DataFrame,
            column: str,
            lvl: str,
            box_idx: int,
            ax: plt.axes,
            texts: List[plt.text],
            record: bool = True
    ) -> None:
        """
        Adds outliers data to records.

        Parameters
        ----------
        subset : pd.DataFrame
            Subset of pandas dataframe.
        column : str
            Name of the numerical column.
        lvl : str
            "Low" or "High" depending if value is lower than Q1 - alpha*(Q3-Q1) or higher than Q3 + alpha*(Q3-Q1)
        box_idx : int
            Boxplot index.
        ax : plt.axes
            Matplotlib axes.
        texts : List[plt.text]
            List of texts added to boxplot.
        record : bool
            If true, we add outliers id and warning message to record, else we only append patients' ids to boxplot.
        """
        for i in range(subset.shape[0]):

            # We extract patient's id and value
            patient, value = subset.iloc[i, 0], subset.iloc[i, 1]
            texts.append(ax.text(box_idx, value, patient))

            if record:
                # We create the appropriate warning message
                warning = self.__return_outlier_warning(column, value, lvl)

                # We add the warning to the records
                self.__update_single_patient_records(patient, warning)

    def __update_records(
            self,
            key: str,
            sub_key: str,
            warning: str
    ) -> None:
        """
        Adds warning message to a patient or a column in the records.

        Parameters
        ----------
        key : str
            Choose between self.CP or self.CC.
        sub_key : str
            Patient or column name.
        warning : str
            Warning message.
        """
        if sub_key in self.__records[key].keys():
            self.__records[key][sub_key].update({len(self.__records[key][sub_key].keys()): warning})
        else:
            self.__records[key][sub_key] = {0: warning}

    def __update_single_patient_records(
            self,
            patient: str,
            warning: str
    ) -> None:
        """
        Adds a warning message to a patient in the records.

        Parameters
        ----------
        patient : str
            Patient ID.
        warning : str
            Warning message
        """
        self.__update_records(self.CP, patient, warning)

    def __update_single_column_records(
            self,
            column: str,
            warning: str
    ) -> None:
        """
        Adds a warning message to a column in the records.

        Parameters
        ----------
        column : str
            Column name.
        warning : str
            Warning message.
        """
        self.__update_records(self.CC, column, warning)

    def __save_records(
            self
    ) -> None:
        """
        Saves the records dictionary into a json file.
        """
        # We save all the removed rows and columns collected in a json file
        with open(join(self.__records_path, "cleaner_record.json"), "w") as file:
            dump(self.__records, file, indent=True)

    @staticmethod
    def __calculate_squared_mahalanobis_distances(
            df: pd.DataFrame,
            numerical_columns: List[str]
    ) -> np.array:
        """
        Computes mahalanobis distances of all patients according to their numerical values.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.
        numerical_columns : Optional[List[str]]
            List with names of numerical columns.

        Returns
        -------
        squared_distances : numpy array
            Mahalanobis distances of all patients according to their numerical values.
        """

        # We fill NaN values with column means and convert the data to numpy array
        numerical_df = df[numerical_columns]
        X = numerical_df.to_numpy().reshape((df.shape[0], -1))

        # We get the centroid (the vector composed with the means of each variable)
        centroid = np.mean(X, axis=0).reshape(1, numerical_df.shape[1])

        # We get the inverse of the covariance matrix
        covariance_inv = np.linalg.matrix_power(np.cov(X, rowvar=False), -1)

        # We compute the first matrix multiplication
        A = (X - centroid)
        B = A.dot(covariance_inv)

        # We finish calculating distances all at once with einsum
        squared_distances = np.einsum('ij,ij->i', B, A)

        return squared_distances

    @staticmethod
    def __extract_quartiles(
            bp: plt.boxplot,
            idx_of_box: int = 0
    ) -> Tuple[float, float, float, float]:
        """
        Extracts quartiles data out of matplotlib boxplot

        Parameters
        ----------
        bp : plt.boxplot
            matplotlib boxplot.
        idx_of_box : int
            Index of the box in the boxplot.

        Returns
        -------
        q1, q2, q3, iqr (q3 - q1) : Tuple[float, float, float, float]
            First quartile, second quartile, third quartile and interquartile range.
        """

        q1 = bp['boxes'][idx_of_box].get_ydata()[1]
        q2 = bp['medians'][idx_of_box].get_ydata()[1]
        q3 = bp['boxes'][idx_of_box].get_ydata()[2]

        return q1, q2, q3, round(q3 - q1, 4)

    def __refactor_dataframe(
            self,
            df: pd.DataFrame,
            numerical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Filters the dataframe to only keep the PARTICIPANT column and specific numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to apply cleaning on.
        numerical_columns : Optional[List[str]]
            List with names of numerical columns.

        Returns
        -------
        updated_df : pd.DataFrame
            Same dataframe with only numerical columns.
        """
        if self.__ids_col not in df.columns.values:
            raise ValueError(f"{self.__ids_col} column must be in the dataframe")

        updated_df = df.copy()
        updated_df[numerical_columns] = df[numerical_columns].astype(float).fillna(df[numerical_columns].mean())

        return updated_df

    @staticmethod
    def __return_categorical_warning(
            category: str,
            value: float,
            min_warning: bool = True
    ) -> str:
        """
        Returns a string to use as a warning message for column in the records.

        Parameters
        ----------
        category : str
            Category of a categorical variable.
        value : float
            Number of times the category appears or percentage of column occupied by the category.
        min_warning : bool
            True if the warning is associated to a low count of category appearances.

        Returns
        -------
        warning : str
            Warning message.
        """
        if min_warning:
            return f"Only {value} items had the category {category}."
        else:
            return f"Category {category} was representing {round(value*100,2)} % of the column."

    @staticmethod
    def __return_outlier_warning(
            numerical_attribute: str,
            value: float,
            lvl: str
    ) -> str:
        """
        Returns a string to use as warning message for the patient in the records.

        Parameters
        ----------
        numerical_attribute : str
            Name of the column in the dataframe.
        value : float
            Float value of the patient.
        lvl : str
            "Low" or "High" depending if value is lower than Q1 - alpha*(Q3-Q1) or higher than Q3 + alpha*(Q3-Q1).

        Returns
        -------
        warning : str
            Warning message.
        """

        return f"Patient targeted as an outlier due to a {lvl} '{numerical_attribute}' value of {value}."

    @staticmethod
    def __return_mahalanobis_warning(
            value: float
    ) -> str:
        """
        Returns a string to use as warning message for patient in the records.

        Parameters
        ----------
        value: float
            Patient's Mahalanobis distance.

        Returns
        -------
        warning : str
            Warning message.
        """
        return f"Patient targeted as an outlier due to its Mahalanobis distance of {value}."
