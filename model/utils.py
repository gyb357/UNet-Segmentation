from typing import Any


@staticmethod
def ternary_op(a: bool, b: Any, c: Any) -> Any:
    """
    Ternary operator implementation.

    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (Any): Value to return if condition is `False`
    """

    return b if a else c

@staticmethod
def ternary_op_elif(a: bool, b: Any, c: bool, d: Any, e: Any) -> Any:
    """
    Ternary operator with `elif` implementation.
    
    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (bool): Condition to check if `a` is `False`
        d (Any): Value to return if `c` is `True`
        e (Any): Value to return if `c` is `False`
    """
    
    return b if a else ternary_op(c, d, e)

