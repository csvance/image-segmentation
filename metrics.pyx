from libc.math cimport log10 as log_base_10
from libc.math cimport log2 as log_base_2

def huber(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double delta = 0.0125
  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])

    if res <= delta:
      res_sum += (res**2)/2
    else:
      res_sum += delta*(res - delta/2)

  return res_sum / n

def l2_log10(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double delta = 0.0125
  cdef double delta_bias = (delta**2)/2
  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])

    if res <= delta:
      res_sum += (res**2)/2
    else:
      res_sum += log_base_10(res*2 + 1)/2 + delta_bias

  return res_sum / n

def log10(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])
    res_sum += log_base_10(res + 1)

  return res_sum / n

def l2_log2(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double delta = 0.0125
  cdef double delta_bias = (delta**2)/2
  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])

    if res <= delta:
      res_sum += (res**2)/2
    else:
      res_sum += log_base_2(res*2 + 1)/2 + delta_bias

  return res_sum / n

def log2(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])
    res_sum += log_base_2(res + 1)

  return res_sum / n