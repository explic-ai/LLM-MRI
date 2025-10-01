from llm_mri import ActivationAreas
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
import pandas as pd
import numpy as np
from typing import Union

class Evaluation:

    def __init__(self, activation_areas:ActivationAreas):
        """
        Initializes the evaluation class with the activation areas class and number of components. 
        If no number of components is specified, the comparison will consider all of the embeddings in the comparison.
        
        activation_areas (ActivationAreas): An instance of the ActivationAreas class
        """
        
        self.activation_areas = activation_areas



    def _subtract_reports(self, report_full, report_reduced):
        """
        Function to return comparison on every metric from the fill report and the reduced report
        """
        diff_report = {}
        for key in report_full:
            # Skip support because it's just a count of samples, not a performance metric
            if key == "support":
                continue
            if isinstance(report_full[key], dict):
                diff_report[key] = {}
                for metric in report_full[key]:
                    if metric == "support":
                        continue
                    diff_report[key][metric] = report_full[key][metric] - report_reduced[key][metric]
            else:
                # For global values like 'accuracy'
                diff_report[key] = report_full[key] - report_reduced[key]
        return diff_report


    def evaluate_model(self, n_splits:int = 5, test_size:float = 0.3, random_state:int = 42, n_components:int = None, metrics:Union[list, str] = None):
        """
        Evaluates the model using the original and reduced embeddings.
        This method will train a classifier with the dataset obtained previously, using the parameters predefined by the user, such as k-fold, train-test split, etc.
        Returns the difference between the metrics of the two classifiers.
        """

        # Obtaining data from embeddings and nrags
        X_reduced, y_reduced = self.activation_areas._get_nrag_embeddings()
        X_full, y_full = self.activation_areas._get_embeddings()

        # Flattening y if necessary
        y_reduced = pd.Series(np.ravel(y_reduced))
        y_full = pd.Series(np.ravel(y_full))

        print(f"Reduced shape: {X_reduced.shape}, Full shape: {X_full.shape}")

        # Tests (remove afterwards)
        if len(X_reduced) != len(X_full):
            raise ValueError("Reduced e Full têm números diferentes de instâncias.")
        if len(y_reduced) != len(y_full):
            raise ValueError("Os vetores de rótulos têm tamanhos diferentes.")
        if not y_reduced.reset_index(drop=True).equals(y_full.reset_index(drop=True)):
            raise ValueError("As ordens/valores de rótulo diferem entre Reduced e Full.")

        # Resetting indices to ensure alignment
        Xr = X_reduced.reset_index(drop=True)
        Xf = X_full.reset_index(drop=True)
        y = y_reduced.reset_index(drop=True)

        # Evaluating metrics passed by user
        if metrics is None:
            metrics = [
                "f1_macro", "f1_weighted",
                "recall_macro", "recall_weighted",
                "accuracy", "balanced_accuracy",
            ]

        elif isinstance(metrics, str):  
            metrics = [metrics]
        
        # Constructing scorers dictionary
        scorers = {m: m for m in metrics}

        # Creating splits
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        splits = list(sss.split(Xr, y))  # list of (train_idx, test_idx)

        # Defining pipelines for each classifier
        pipe_reduced = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=1000, random_state=random_state))
        pipe_full    = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=1000, random_state=random_state))

        # Creating cross validation for each classifier
        cv_reduced = cross_validate(pipe_reduced, Xr, y, cv=splits, scoring=scorers, return_estimator=False)
        cv_full    = cross_validate(pipe_full,    Xf, y, cv=splits, scoring=scorers, return_estimator=False)

        # Calculating difference between metrics, and displaying on the delta dictionary
        delta = {}
        for m in metrics:
            r = np.asarray(cv_reduced[f"test_{m}"])
            f = np.asarray(cv_full[f"test_{m}"])
            d = f - r
            delta[m] = {
                "per_split": d.tolist(),
                "mean": float(d.mean()),
            }

        return {"delta": delta}

