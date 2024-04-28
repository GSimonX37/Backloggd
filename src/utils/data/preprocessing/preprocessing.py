import pandas as pd

from .preparation import spaces
from .preparation import length
from .preparation import letters
from .lemmatization import lemmatize


def preprocessing(data: pd.Series) -> pd.Series:
    # Очистка текста
    data = data.apply(letters)
    data = data.apply(length, size=2)
    data = data.apply(spaces)

    # Лемматизация текста.
    data = data.apply(lemmatize)

    # Приведение текста к нижнему регистру.
    data = data.str.lower()

    return data
