import math, random
import unicodedata
import re
import torch

def unicodeToAscii(s: str) -> str:
    """
    Convert a Unicode string to ASCII, if possible.
    Args:
        s: unicode string.
    Returns:

    """
    ascii_string =  ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    return ascii_string

def normalizeString(s: str) -> str:
    """
    Normalize a string.
    Args:
        s: string.
    Returns:
        s: normalized string.
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s