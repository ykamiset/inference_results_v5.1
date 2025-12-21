# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import platform
import subprocess
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from nvmitten.utils import run_command

import logging
logging.basicConfig(level=(logging.INFO if os.environ.get("VERBOSE", '0') == '0' else logging.DEBUG),
                    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

from .paths import VERSION_FILE


with VERSION_FILE.absolute().open(mode='r') as fh:
    __MLPERF_INF_VERSION__ = fh.readline().strip('\n')
__MLPERF_INF_PAST_VERSIONS__ = ["v5.0", "v4.1", "v4.0", "v3.1", "v3.0", "v2.1", "v2.0", "v1.1", "v1.0", "v0.7", "v0.5"]


def args_to_string(d, blacklist=[], delimit=True, double_delimit=False):
    flags = []
    for flag in d:
        # Skip unset
        if d[flag] is None:
            continue
        # Skip blacklisted
        if flag in blacklist:
            continue
        if type(d[flag]) is bool:
            if d[flag] is True:
                flags.append("--{:}=true".format(flag))
            elif d[flag] is False:
                flags.append("--{:}=false".format(flag))
        elif type(d[flag]) in [int, float] or not delimit:
            flags.append("--{:}={:}".format(flag, d[flag]))
        else:
            if double_delimit:
                flags.append("--{:}=\\\"{:}\\\"".format(flag, d[flag]))
            else:
                flags.append("--{:}=\"{:}\"".format(flag, d[flag]))
    return " ".join(flags)


def flags_bool_to_int(d):
    for flag in d:
        if type(d[flag]) is bool:
            if d[flag]:
                d[flag] = 1
            else:
                d[flag] = 0
    return d


def dict_get(d, key, default=None):
    """Return non-None value for key from dict. Use default if necessary."""

    val = d.get(key, default)
    return default if val is None else val


def dict_eq(d1: Dict[str, Any], d2: Dict[str, Any], ignore_keys: Optional[Set[str]] = None) -> bool:
    """Compares 2 dictionaries, returning whether or not they are equal. This function also supports ignoring keys for
    the equality check. For example, if d1 is {'a': 1, 'b': 2} and d2 is {'a': 1, 'b': 3, 'c': 1}, if ignore_keys is set
    to {'b', 'c'}, this method will return True.
    While this method supports dicts with any type of keys, it is recommended to use strings as keys.

    Args:
        d1 (Dict[str, Any]): The first dict to be compared
        d2 (Dict[str, Any]): The second dict to be compared
        ignore_keys (Set[str]): If set, will ignore keys in this set when doing the equality check

    Returns:
        bool: Whether or not d1 and d2 are equal, ignore the keys in `ignore_keys`
    """
    def filter_dict(d): return {k: v for k, v in d.items() if k not in ignore_keys}
    return filter_dict(d1) == filter_dict(d2)
