"""
    @file:              json_encoder.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the `EnhancedJSONEncoder` used within the `TuningRecorder`.
"""

from dataclasses import asdict, is_dataclass
from json import JSONEncoder

from optuna.trial import BaseTrial
from optuna.storages import BaseStorage
from optuna.samplers import BaseSampler
from optuna.study import Study


class EnhancedJSONEncoder(JSONEncoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["default"] = str
        self.encoder = JSONEncoder(**kwargs)

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        elif isinstance(o, (BaseSampler, BaseTrial)):
            return {k: self.default(v) for k, v in vars(o).items()}
        elif isinstance(o, Study):
            d = {}
            for k, v in vars(o).items():
                if isinstance(v, BaseStorage):
                    d[k] = o.user_attrs
                else:
                    d[k] = self.default(v)
            return d
        elif isinstance(o, dict):
            return {k: self.default(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.default(v) for v in o]

        return self.encoder.default(o)
