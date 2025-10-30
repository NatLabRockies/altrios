from importlib.metadata import version

__version__ = version("altrios")

import numpy as np
from pathlib import Path
import re
from typing import List, Union, Dict, Optional, Any
from typing_extensions import Self
import pandas as pd
import polars as pl

from altrios.loaders.powertrain_components import _res_from_excel
from altrios.utilities import copy_demo_files  # noqa: F401
from altrios.utilities import package_root, resources_root  # noqa: F401
from altrios import utilities as utils  # noqa: F401

# make everything in altrios_pyo3 available here
from altrios.altrios_pyo3 import *  # noqa: F403
from altrios import *  # noqa: F403


def __array__(self):
    return np.array(self.tolist())


# creates a list of all python classes from rust structs that need python-side serde helpers
ACCEPTED_RUST_STRUCTS = [
    attr
    for attr in altrios_pyo3.__dir__()
    if not attr.startswith("__")
    and isinstance(getattr(altrios_pyo3, attr), type)  # noqa: F405
    and attr[0].isupper()
]

# TODO connect to crate features
data_formats = [
    "yaml",
    "msg_pack",
    # 'toml',
    "json",
]


def to_pydict(self, data_fmt: str = "msg_pack", flatten: bool = False) -> Dict:
    """
    Returns self converted to pure python dictionary with no nested Rust objects
    # Arguments
    - `flatten`: if True, returns dict without any hierarchy
    - `data_fmt`: data format for intermediate conversion step
    """
    data_fmt = data_fmt.lower()
    assert data_fmt in data_formats, f"`data_fmt` must be one of {data_formats}"
    match data_fmt:
        case "msg_pack":
            import msgpack  # type: ignore[import-untyped]

            pydict = msgpack.loads(self.to_msg_pack())
        case "yaml":
            from yaml import load  # type: ignore[import-untyped]

            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            pydict = load(self.to_yaml(), Loader=Loader)
        case "json":
            from json import loads

            pydict = loads(self.to_json())

    if not flatten:
        return pydict
    else:
        hist_len = get_hist_len(pydict)
        assert hist_len is not None, "Cannot be flattened"
        flat_dict = get_flattened(pydict, hist_len)
        return flat_dict


def get_hist_len(obj: Dict) -> Optional[int]:
    """
    Finds nested `history` and gets lenth of first element
    """
    # TODO: check if this is sufficiently recursive and if it's not, make it recursive all the way down

    if "history" in obj.keys():
        return len(next(iter(obj["history"].values())))

    elif (
        next(
            iter(
                k for k in obj.keys() if re.search("(history\\.\\w+)$", k) is not None
            ),
            None,
        )
        is not None
    ):
        return len(
            next(
                (
                    v
                    for (k, v) in obj.items()
                    if re.search("(history\\.\\w+)$", k) is not None
                )
            )
        )

    for k, v in obj.items():
        if isinstance(v, dict):
            hist_len = get_hist_len(v)
            if hist_len is not None:
                return hist_len
    return None


def get_flattened(obj: Dict | List, hist_len: int, prepend_str: str = "") -> Dict:
    """
    Flattens and returns dictionary, separating keys and indices with a `"."`
    # Arguments
    # - `obj`: object to flatten
    # -  hist_len: length of any lists storing history data
    # - `prepend_str`: prepend this to all keys in the returned `flat` dict
    """
    flat: Dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = k if (prepend_str == "") else prepend_str + "." + k
            if isinstance(v, dict) or (isinstance(v, list) and len(v) != hist_len):
                flat.update(get_flattened(v, hist_len, prepend_str=new_key))
            else:
                flat[new_key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = i if (prepend_str == "") else prepend_str + "." + f"[{i}]"
            if isinstance(v, dict) or (isinstance(v, list) and len(v) != hist_len):
                flat.update(get_flattened(v, hist_len, prepend_str=new_key))
            else:
                flat[new_key] = v
    else:
        raise TypeError("`obj` should be `dict` or `list`")

    return flat


@classmethod  # type: ignore[misc]
def from_pydict(
    cls, pydict: Dict, data_fmt: str = "msg_pack", skip_init: bool = False
) -> Self:  # type: ignore[misc]
    """
    Instantiates Self from pure python dictionary
    # Arguments
    - `pydict`: dictionary to be converted to ALTRIOS object
    - `data_fmt`: data format for intermediate conversion step
    - `skip_init`: passed to `SerdeAPI` methods to control whether initialization
      is skipped
    """
    data_fmt = data_fmt.lower()
    assert data_fmt in data_formats, f"`data_fmt` must be one of {data_formats}"
    match data_fmt.lower():
        case "yaml":
            import yaml

            obj = cls.from_yaml(yaml.dump(pydict), skip_init=skip_init)
        case "msg_pack":
            import msgpack

            obj = cls.from_msg_pack(msgpack.packb(pydict), skip_init=skip_init)
        case "json":
            from json import dumps

            obj = cls.from_json(dumps(pydict), skip_init=skip_init)

    return obj


def to_dataframe(
    self, pandas: bool = False, allow_partial: bool = False
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Returns time series results from ALTRIOS object as a Polars or Pandas dataframe.

    # Arguments
    - `pandas`: returns pandas dataframe if True; otherwise, returns polars dataframe by default
    - `allow_partial`: tries to return dataframe of length equal to solved time steps if simulation fails early
    """
    obj_dict = self.to_pydict(flatten=True)
    history_keys = ["history.", "cyc."]
    hist_len = get_hist_len(obj_dict)
    assert hist_len is not None

    history_dict: Dict[str, Any] = {}
    for k, v in obj_dict.items():
        hk_in_k = any(hk in k for hk in history_keys)
        if hk_in_k and ("__len__" in dir(v)):
            if (len(v) == hist_len) or allow_partial:
                history_dict[k] = v

    if allow_partial:
        cutoff = min([len(val) for val in history_dict.values()])

        if not pandas:
            try:
                df = pl.DataFrame(
                    {col: val[:cutoff] for col, val in history_dict.items()}
                )
            except Exception as err:
                raise Exception(f"{err}\n`save_interval` may not be uniform")
        else:
            try:
                df = pd.DataFrame(
                    {col: val[:cutoff] for col, val in history_dict.items()}
                )
            except Exception as err:
                raise Exception(f"{err}\n`save_interval` may not be uniform")

    else:
        if not pandas:
            try:
                df = pl.DataFrame(history_dict)
            except Exception as err:
                raise Exception(
                    f"{err}\nTry passing `allow_partial=True` to `to_dataframe` or checking for consistent save intervals"
                )
        else:
            try:
                df = pd.DataFrame(history_dict)
            except Exception as err:
                raise Exception(
                    f"{err}\nTry passing `allow_partial=True` to `to_dataframe` or checking for consistent save intervals"
                )
    return df


# adds variable_path_list() and history_path_list() as methods to all classes in
# ACCEPTED_RUST_STRUCTS
for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(altrios_pyo3, item), "to_pydict", to_pydict)  # noqa: F405
    setattr(getattr(altrios_pyo3, item), "from_pydict", from_pydict)  # noqa: F405
    setattr(getattr(altrios_pyo3, item), "to_dataframe", to_dataframe)  # noqa: F405

setattr(ReversibleEnergyStorage, "from_excel", classmethod(_res_from_excel))  # noqa: F405
