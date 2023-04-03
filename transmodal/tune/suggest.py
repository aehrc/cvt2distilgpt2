from ray.tune.suggest.hebo import HEBOSearch
from typing import Dict, Optional
import numpy as np

class ModifiedHEBOSearch(HEBOSearch):

    intermediate_results = {}

    def on_trial_result(self, trial_id: str, result: Dict):
        self.intermediate_results.setdefault(trial_id, []).append(result)

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        if result:
            if self._mode == "max":
                idx = np.argmax([i[self._metric] for i in self.intermediate_results[trial_id]])
            elif self._mode == "min":
                idx = np.argmin([i[self._metric] for i in self.intermediate_results[trial_id]])
            self._process_result(trial_id, self.intermediate_results[trial_id][idx])
            self.intermediate_results.pop(trial_id, None)
        self._live_trial_mapping.pop(trial_id)