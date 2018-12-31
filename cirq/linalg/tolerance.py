# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility methods for testing approximate equality of matrices and scalars within specified tolerances."""

import numpy as np

DEFAULT_RTOL: float = 1e-5
DEFAULT_ATOL: float = 1e-8
DEFAULT_EQUAL_NAN: bool = False


def all_close(a, b, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
              equal_nan: bool = DEFAULT_EQUAL_NAN) -> bool:
    """Returns whether the given matrices are approximately equal within the specified tolerance parameters.

    Args:
        a: First matrix to compare.
        b: Second matrix to compare.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether to compare NaN's as equal.
    """
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def all_near_zero(a, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                  equal_nan: bool = DEFAULT_EQUAL_NAN) -> bool:
    """Returns True if the given matrix approximately contains all zero elements.

    Args:
        a: Matrix to evaluate.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether to compare NaN's as equal.
    """
    return all_close(a, np.zeros(np.shape(a)), rtol, atol, equal_nan)


def all_near_zero_mod(a, period, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                      equal_nan: bool = DEFAULT_EQUAL_NAN):
    return all_close((np.array(a) + (period / 2)) % period - period / 2,
                     np.zeros(np.shape(a)), rtol, atol, equal_nan)


def close(a, b, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL):
    return abs(a - b) <= atol + rtol * abs(b)


def near_zero(a, atol: float = DEFAULT_ATOL):
    return abs(a) <= atol


def near_zero_mod(a, period, atol: float = DEFAULT_ATOL):
    half_period = period / 2
    return near_zero((a + half_period) % period - half_period, atol)


class Tolerance:
    """Specifies thresholds for doing approximate equality."""

    ZERO = None  # type: Tolerance
    DEFAULT = None  # type: Tolerance

    def __init__(self,
                 rtol: float = 1e-5,
                 atol: float = 1e-8,
                 equal_nan: bool = False) -> None:
        """Initializes a Tolerance instance with the specified parameters.

        Notes:
          Matrix Comparisons (methods beginning with "all_") are done by
          numpy.allclose, which considers x and y
          to be close when abs(x - y) <= atol + rtol * abs(y). See
          numpy.allclose's documentation for more details.   The scalar
          methods perform the same calculations without the numpy
          matrix construction.

        Args:
          rtol: Relative tolerance.
          atol: Absolute tolerance.
          equal_nan: Whether NaNs are equal to each other.
        """
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    # Matrix methods
    def all_close(self, a, b):
        return all_close(a, b, self.rtol, self.atol)

    def all_near_zero(self, a):
        return all_near_zero(a, self.rtol, self.atol, self.equal_nan)

    def all_near_zero_mod(self, a, period):
        return all_near_zero_mod(a, period, self.rtol, self.atol, self.equal_nan)

    # Scalar methods
    def close(self, a, b):
        return close(a, b, atol=self.atol, rtol=self.rtol)

    def near_zero(self, a):
        return near_zero(a, atol=self.atol)

    def near_zero_mod(self, a, period):
        return near_zero_mod(a, period, atol=self.atol)

    def __repr__(self):
        return "Tolerance(rtol={}, atol={}, equal_nan={})".format(
            repr(self.rtol), repr(self.atol), repr(self.equal_nan))


Tolerance.ZERO = Tolerance(rtol=0, atol=0)
Tolerance.DEFAULT = Tolerance()
