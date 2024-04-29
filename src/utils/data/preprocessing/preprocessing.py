import pandas as pd

from .cleaning import length
from .cleaning import letters
from .cleaning import spaces
from .lemmatization import lemmatize


def preprocess(data: pd.Series) -> pd.Series:
    # Очистка текста
    data = data.apply(letters)
    data = data.apply(length, size=2)
    data = data.apply(spaces)

    # Лемматизация текста.
    data = data.apply(lemmatize)

    # Удаление записей, с длиной текста меньше 50 символов.
    data = data[data.str.len() >= 50]

    # Приведение текста к нижнему регистру.
    data = data.str.lower()

    return data
