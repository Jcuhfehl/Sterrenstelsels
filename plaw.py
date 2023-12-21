import numpy.random

def plaw(n, xmin, xmax, a):
    norm = (xmax/xmin)**(1.+a) - 1.
    mi = xmin * (norm * numpy.random.random(n) + 1.)**(1./(1.+a))
    return mi
