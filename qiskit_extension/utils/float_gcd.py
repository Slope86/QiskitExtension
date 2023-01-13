import math
from functools import reduce


def float_gcd(*numbers: float) -> float:
    """Calculate the greatest common divisor of a list of floats.

    Example:
        >>> float_gcd(0.75, 0.25, 0.375)
        0.125

    Args:
        numbers (float): A list of floats.

    Returns:
        float: The greatest common divisor of the list of floats.
    """
    return reduce(_gcd, numbers)


def _gcd(a: float, b: float) -> float:
    # Make sure that a >= b
    if a < b:
        return _gcd(b, a)

    # If b is close to 0, then a is the answer
    if abs(b) < 0.001:
        return a
    # If b is not close to 0, then the answer is the GCD of b and the remainder of a/b
    else:
        return _gcd(b, a - math.floor(a / b) * b)


if __name__ == "__main__":
    print(float_gcd(0.75, 0.25, 0.375))
    print(float_gcd(15, 20))
