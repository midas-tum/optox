import numpy as np

import _ext.py_scale_operator

def forward(x):
    if x.dtype == np.float32:
        return _ext.py_scale_operator.Scale_float().forward(x)
    elif x.dtype == np.float64:
        return _ext.py_scale_operator.Scale_double().forward(x)
    elif x.dtype == np.complex64:
        return _ext.py_scale_operator.Scale_float2().forward(x)
    elif x.dtype == np.complex128:
        return _ext.py_scale_operator.Scale_double2().forward(x)
    else:
        raise RuntimeError('Unsupported dtype: {}'.format(x.dtype))
    