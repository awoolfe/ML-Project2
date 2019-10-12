from typing import Tuple, Dict, List


def preprocess_text(text: str) -> Tuple[List[str], Dict]:
    """Clean and tokenize text before feature creation.

    Returns list of tokens.
    Also return meta dict with information about preprocessing
    (e.g. tokens removed, length of original text, etc...)
    """
    raise NotImplementedError
