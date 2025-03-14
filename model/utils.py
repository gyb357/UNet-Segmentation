from typing import Any


def ternary_op(a: bool, b: Any, c: Any) -> Any:
    return b if a else c

def ternary_op_elif(a: bool, b: Any, c: bool, d: Any, e: Any) -> Any:
    return b if a else ternary_op(c, d, e)

