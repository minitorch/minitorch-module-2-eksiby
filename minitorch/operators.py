"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

def mul(a, b):
    return a * b

def id(a):
    return a

def add(a, b):
    return a + b

def neg(a):
    return -a

def lt(a, b):
    return a < b

def eq(a, b):
    return a == b

def max(a, b):
    return b if lt(a, b) else a

def is_close(a, b):
    return abs(a - b) < 1e-2

def sigmoid(a):
    return 1.0/ (1.0 + exp(-a)) 

def relu(a):
    return max(0.0, a)

def log(a):
    return math.log(a)

def exp(a):
    return math.exp(a)

def inv(a):
    return 1.0 / a

def log_back(a, b):
    return 1.0 / a * b

def inv_back(a, b):
    return - inv(a)**2 * b

def relu_back(a, b):
    return b if a > 0.0 else 0.0
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
def map(func, a):
    out = []
    for el in a:
        out += [func(el)]
    return out
def zipWith(func, a, b):
    out = []
    for i in range(min(len(a), len(b))):
        out += [func(a[i], b[i])]
    return out

def reduce(func, a):
    if len(a) == 0:
        return 0
    out = a[0]
    for i in range(1, len(a)):
        out = func(out, a[i])
    return out


def negList(a):
    return map(lambda x: -x, a)

def addLists(a, b):
    return zipWith(lambda x, y: x + y, a, b)

def sum(a):
    return reduce(lambda x, y: x + y, a)

def prod(a):
    return reduce(lambda x, y: x * y, a)

# TODO: Implement for Task 0.3.
