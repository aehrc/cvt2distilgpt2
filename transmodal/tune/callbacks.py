from ray.tune import Callback
from typing import List
import gc

class GarbageCollection(Callback):
    def on_trial_complete(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        **info,
    ):
        gc.collect()
