import math
import os
import sys

import time

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Get the absolute path to the directory where figa.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# One levels up from figa.py
sys.path.append(os.path.join(base_path, "../"))

from figa_tutorial.ga import GA
from utils.helpers import get_experiment_params, generate_report, get_data

from sklearn.base import BaseEstimator, ClassifierMixin


class FIGA:
    def __init__(self, config, model, sensitive_param, population_size=200, cluster_num=4, threshold=0):
        self.cluster_num = cluster_num
        self.binary_threshold = threshold
        self.config = config
        self.threshold = threshold
        self.population_size = population_size

        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.input_bounds = np.array(self.config.input_bounds)
        self.sensitive_param = sensitive_param
        self.model = model

        self.start_time = time.time()
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.tracking_interval = 1000

        self.load_data()

    def load_data(self):
        data = get_data(self.config.dataset_name)
        X, Y, self.input_shape, self.nb_classes = data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def evaluate_disc(self, inp):
        inp0 = np.array([int(x) for x in inp])
        inp1 = np.array([int(x) for x in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        inp1 = inp1.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        self.total_generated += 1

        if tuple(map(tuple, inp0)) not in self.disc_inputs:
            out0 = self.make_prediction(inp0)

            sensitive_values = range(self.config.input_bounds[self.sensitive_param - 1][0],
                                     self.config.input_bounds[self.sensitive_param - 1][1] + 1)

            for i in sensitive_values:
                if i != original_sensitive_value:
                    inp1[0][self.sensitive_param - 1] = i
                    out1 = self.make_prediction(inp1)

                    if abs(out0 - out1) > self.threshold:
                        self.disc_inputs.add(tuple(map(tuple, inp0)))
                        self.disc_inputs_list.append(inp0.tolist()[0])

                        self.set_time_to_1000_disc()

                        return 1

        return 0

    def make_prediction(self, inp):
        output = self.model.predict(inp)

        return (output > 0.5).astype(int)

    def get_feature_importance(self):

        if self.model.__class__.__name__ == 'SVC':
            # use only 100 samples from both X_test and y_test
            X_test = self.X_test[:200]
            y_test = self.y_test[:200]
        else:
            # use a maximum of 1000 samples from both X_test and y_test
            X_test = self.X_test[:1000]
            y_test = self.y_test[:1000]

        if self.model is None:
            raise ValueError("Model is None; ensure the correct model is loaded.")
        result = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42)

        feature_importance = result.importances_mean / np.sum(result.importances_mean)
        return feature_importance

    def update_cumulative_efficiency(self, iteration):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):
        self.start_time = time.time()

        feature_importance = self.get_feature_importance()


        ga = GA(
            pop_size=self.population_size,
            dna_size=len(self.config.input_bounds),
            bound=self.config.input_bounds,
            fitness_func=self.evaluate_disc,
            feature_importance=feature_importance
        )

        count = 300

        max_evolution = math.ceil(max_samples / self.population_size)

        for i in range(max_evolution):
            ga.evolve()
            self.update_cumulative_efficiency(i)

            use_time = time.time() - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        generate_report(
            approach_name='FIGA',
            dataset_name=self.config.dataset_name,
            classifier_name=self.model.__class__.__name__,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
    model = joblib.load(classifier_path)

    figa = FIGA(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    figa.run(max_allowed_time=max_allowed_time)
