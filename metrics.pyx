from libc.math cimport log as log_n

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

def l2_log(double[:] x, double[:] y):

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
      res_sum += log_n(res*2 + 1)/2 + delta_bias

  return res_sum / n

def log_10(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])
    res_sum += log_n(res + 1)

  return res_sum / n