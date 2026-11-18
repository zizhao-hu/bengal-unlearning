from evals.base import Evaluator


class HubbleEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("HUBBLE", eval_cfg, **kwargs)
