"""
Helper utilities for the job search application
"""

import re


def inline_sql(query: str, params: list) -> str:
    """
    Human-readable SQL with params inlined for debugging.
    Do NOT execute the returned SQL.
    """
    def q(v):
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

    out = query
    for i, v in enumerate(params, start=1):
        if isinstance(v, (list, tuple)):
            rep = "ARRAY[" + ", ".join(q(x) for x in v) + "]"
        else:
            rep = q(v)
        pattern = re.compile(rf"\${i}(?!\d)")
        out = pattern.sub(rep, out)
    return out
