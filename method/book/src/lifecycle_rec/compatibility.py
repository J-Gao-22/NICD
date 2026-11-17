import importlib
import sys
import numpy as np
try:
    importlib.import_module('numpy._core')
except ModuleNotFoundError:
    sys.modules['numpy._core'] = np.core
    for name in ('numeric', 'multiarray', '_multiarray_umath'):
        sys.modules['numpy._core.' + name] = importlib.import_module('numpy.core.' + name)
