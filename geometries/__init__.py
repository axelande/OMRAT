
def isint(x: str) -> bool:
    """Checks if the inserted value is of int type

    Parameters
    ----------
    x: str

    Returns
    -------
    bool
    """
    try:
        a = float(x)
        b = int(a)
    except (ValueError, OverflowError, TypeError):
        return False
    else:
        return a == b