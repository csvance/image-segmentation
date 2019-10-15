import math

def huber(double[:] x, double[:] y):

  cdef int n = x.shape[0]

  cdef double delta = 0.025
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

  cdef double delta = 0.025
  cdef double res = 0
  cdef double res_sum = 0

  for i in range(n):
    res = abs(x[i] - y[i])

    if res <= delta:
      res_sum += (res**2)/2
    else:
      res_sum += math.log(res*2 + 1) + (delta**2)/2

  return res_sum / n