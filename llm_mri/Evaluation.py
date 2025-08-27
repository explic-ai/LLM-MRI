from llm_mri import ActivationAreas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

class Evaluation:

    def __init__(self, activation_areas:ActivationAreas, n_components:int = None):
        """
        Initializes the evaluation class with the activation areas class and number of components. 
        If no number of components is specified, the comparison will consider all of the embeddings in the comparison.
        
        activation_areas (ActivationAreas): An instance of the ActivationAreas class
        n_components (int): The number of components to consider in the evaluation. If set to 0, all components (original embedding) will be used.
        """
        
        self.activation_areas = activation_areas
        self.n_components = n_components

    def _get_reduced_embeddings(self):
        """
        Obtains the hidden states, reduced to the number of components specified in the constructor.
        """
        pass

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

    def evaluate_model(self, n_splits:int = 5, test_size:float = 0.3, random_state:int = 42):
        # Treina um classificador com o dataset obtido previamente, utilizando os parâmetros pré-definidos pelo usuário, como k-fold, split treino e teste, etc.
        # Retorna as métricas de acordo com o sklearn
        """
        Evaluates the model using the activation areas and reduced embeddings.
        This method will train a classifier with the dataset obtained previously, using the parameters predefined by the user, such as k-fold, train-test split, etc.
        Returns the metrics according to sklearn.
        """

        # Obtaining reduced hidden states
        reduced_hidden_states, y_reduced = self.activation_areas.get_nrag_embeddings(self.n_components)

        # Obtaining full embeddings for comparison effects
        full_embeddings, y_embeddings = self.activation_areas.get_nrag_embeddings()

        # Adjusting shape to 1 dimension only
        y_reduced = y_reduced.squeeze()
        y_embeddings = y_embeddings.squeeze()

        # Training two different classifiers with both datasets, using the folds number, test_size, and random_state parameters.
        #dictionary to store results
        results = {}

        #train/test split for both datasets (keeps stratification)
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            reduced_hidden_states, y_reduced, test_size=test_size,
            random_state=random_state, stratify=y_reduced
        )
        Xf_train, Xf_test, yf_train, yf_test = train_test_split(
            full_embeddings, y_embeddings, test_size=test_size,
            random_state=random_state, stratify=y_embeddings
        )

        #define classifiers
        clf_reduced = LogisticRegression(max_iter=500, random_state=random_state)
        clf_full = LogisticRegression(max_iter=500, random_state=random_state)

        #fit classifiers
        clf_reduced.fit(Xr_train, yr_train)
        clf_full.fit(Xf_train, yf_train)

        #predictions on test sets
        yr_pred = clf_reduced.predict(Xr_test)
        yf_pred = clf_full.predict(Xf_test)

        #classification reports (sklearn metrics)
        reduced_classification_report = classification_report(yr_test, yr_pred, output_dict=True)
        full_classification_report = classification_report(yf_test, yf_pred, output_dict=True)
        
        # obtained subtracted metrics from the model trained with full dataset and the model trained with reduced dataset
        results['report_difference'] = self._subtract_reports(full_classification_report, reduced_classification_report)

        #cross-validation with StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        reduced_cv_scores = cross_val_score(clf_reduced, reduced_hidden_states, y_reduced, cv=skf, scoring="f1_macro").mean()
        full_cv_scores = cross_val_score(clf_full, full_embeddings, y_embeddings, cv=skf, scoring="f1_macro").mean()
        results['f1_score_difference'] = full_cv_scores - reduced_cv_scores

        #return dictionary with all metrics
        return results
