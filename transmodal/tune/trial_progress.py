from ray.tune.trial import DEBUG_PRINT_INTERVAL
from ray.tune.logger import pretty_print
from ray.tune.progress_reporter import TrialProgressCallback
from ray.tune.utils.log import Verbosity, has_verbosity
from typing import Dict, List
import os
import time


class ModifiedTrialProgressCallback(TrialProgressCallback):
    def on_trial_complete(self, iteration: int, trials: List["Trial"],
                          trial: "Trial", **info):
        # Only log when we never logged that a trial was completed
        if trial not in self._completed_trials:
            self._completed_trials.add(trial)

            print_result_str = self._print_result(trial.last_result)
            last_result_str = self._last_result_str.get(trial, "")
            # If this is a new result, print full result string
            if print_result_str != last_result_str:
                self.log_result(trial, trial.last_result, error=False)
            else:
                print(f"Trial {trial} completed.")

    def log_result(self, trial: "Trial", result: Dict, error: bool = False):
        done = result.get("done", False) is True
        last_print = self._last_print[trial]
        if done and trial not in self._completed_trials:
            self._completed_trials.add(trial)
        if has_verbosity(Verbosity.V3_TRIAL_DETAILS) and \
           (done or error or time.time() - last_print > DEBUG_PRINT_INTERVAL):
            print("Result for {}:".format(trial))
            print("  {}".format(pretty_print(result).replace("\n", "\n  ")))
            self._last_print[trial] = time.time()
        elif has_verbosity(Verbosity.V2_TRIAL_NORM) and (
                done or error
                or time.time() - last_print > DEBUG_PRINT_INTERVAL):
            info = ""
            if done:
                info = " This trial completed."

            metric_name = self._metric or "_metric"
            metric_value = result.get(metric_name, -99.)

            print_result_str = self._print_result(result)

            self._last_result_str[trial] = print_result_str

            error_file = os.path.join(trial.logdir, "error.txt")

            if error:
                message = f"The trial {trial} errored with " \
                          f"parameters={trial.config}. " \
                          f"Error file: {error_file}"
            elif self._metric:
                message = f"Trial {trial} reported, iter={result['training_iteration']} " \
                          f"{metric_name}={metric_value:.2f} " \
                          f"with parameters={trial.config}.{info}"
            else:
                message = f"Trial {trial} reported, iter={result['training_iteration']} " \
                          f"{print_result_str} " \
                          f"with parameters={trial.config}.{info}"

            print(message)
            self._last_print[trial] = time.time()