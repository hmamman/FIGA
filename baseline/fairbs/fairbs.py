import os
import random
import sys

import joblib
import time
import numpy as np
import pandas as pd
import dice_ml
import json
# Get the absolute path to the directory where fairbs.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairbs.py
sys.path.append(os.path.join(base_path, "../../"))

from utils.helpers import get_experiment_params, generate_report, get_data

class FairBS:
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
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

        self.exp = self.get_explainer()

    def get_explainer(self):
        features = {f'{f}': r for f, r in zip(self.config.feature_name, self.config.input_bounds)}
        d = dice_ml.Data(features=features, continuous_features=[], outcome_name='y')
        backend = 'sklearn'

        model = self.model

        m = dice_ml.Model(model=model, backend=backend)
        method = "random"
        return dice_ml.Dice(d, m, method=method)

    def generate_counterfactuals(self, inp, cf_limit, desired_class='opposite'):
        # query_instance = self.inp_to_df(inp)
        # features_to_vary = [self.config.feature_name[i] for i in range(len(self.config.feature_name)) if
        #                     i is not self.sensitive_param - 1]
        #
        # dice_exp = self.exp.generate_counterfactuals(
        #     query_instance,
        #     total_CFs=cf_limit,
        #     features_to_vary=features_to_vary,
        #     desired_class=desired_class)
        #
        # return json.loads(dice_exp.to_json())['cfs_list'][0]
        try:
            query_instance = self.inp_to_df(inp)
            features_to_vary = [self.config.feature_name[i] for i in range(len(self.config.feature_name)) if
                                i is not self.sensitive_param - 1]

            dice_exp = self.exp.generate_counterfactuals(
                query_instance,
                total_CFs=cf_limit,
                features_to_vary=features_to_vary,
                desired_class=desired_class)

            return json.loads(dice_exp.to_json())['cfs_list'][0]
        except:
            pass
        return []

    def inp_to_df(self, inp):
        return pd.DataFrame(inp, columns=self.config.feature_name)

    def make_prediction(self, inp):
        output = self.model.predict(inp)[0]

        return (output > 0.5).astype(int)

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

    def non_seed_discovery(self, non_seed_limit=1000):
        no_seed_inputs = []
        random.seed(time.time())
        inputs = [
            np.array([random.randint(low, high) for [low, high] in self.input_bounds]) for _ in range(non_seed_limit)
        ]

        for inp in inputs:
            if not self.check_is_seed(inp):
                no_seed_inputs.append(inp)

        return no_seed_inputs

    def counterfactual_discovery(self, inp, cf_limit=1000):
        inp0 = inp.reshape(1, -1)

        cfs = self.generate_counterfactuals(inp0, cf_limit=cf_limit, desired_class='opposite')
        for x in cfs:
            self.evaluate_counterfactual(x[:-1])

    def check_is_seed(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        self.tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1

        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.disc_inputs.add(tuple(map(tuple, inp0)))
                    self.disc_inputs_list.append(inp0.tolist()[0])
                    self.set_time_to_1000_disc()
                    return True
        return False

    def evaluate_counterfactual(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]
        inp0 = inp0.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1

        if tuple(map(tuple, inp0)) in self.disc_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.disc_inputs.add(tuple(map(tuple, inp0)))
                    self.disc_inputs_list.append(inp0.tolist()[0])
                    self.set_time_to_1000_disc()
                    return 1
        return 0

    def run(self, max_global=1000, max_local=1000, max_allowed_time=3600):
        self.start_time = time.time()

        no_seed_inputs = self.non_seed_discovery(non_seed_limit=max_global)

        count = 300

        i = 0

        for inp in no_seed_inputs:
            self.counterfactual_discovery(inp=inp, cf_limit=max_local)
            self.update_cumulative_efficiency(iteration=i)

            use_time = time.time() - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time or self.total_generated >= max_local*max_global:
                break

            i += 1

        elapsed_time = time.time() - self.start_time
        #
        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        generate_report(
            approach_name='FairBS',
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

    experiment = FairBS(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    experiment.run(max_allowed_time=max_allowed_time)
