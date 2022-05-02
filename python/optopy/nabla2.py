import _ext.py_nabla_operator

__all__ = ['float_2d',
           'double_2d',
           'float_3d',
           'double_3d',
           'float_4d',
           'double_4d']

float_2d = _ext.py_nabla_operator.Nabla2_2d_float
double_2d = _ext.py_nabla_operator.Nabla2_2d_double

float_3d = _ext.py_nabla_operator.Nabla2_3d_float
double_3d = _ext.py_nabla_operator.Nabla2_3d_double

float_4d = _ext.py_nabla_operator.Nabla2_4d_float
double_4d = _ext.py_nabla_operator.Nabla2_4d_double