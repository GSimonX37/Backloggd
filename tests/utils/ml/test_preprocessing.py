import pandas as pd
import pytest

from src.utils.ml.preprocessing import cleaning
from src.utils.ml.preprocessing import clear
from src.utils.ml.preprocessing import lemmatization
from src.utils.ml.preprocessing import lemmatize


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', ''),
        ('One. Two. Three. Four. Five.', 'one two three four five'),
        ('One - 1. Two - 2. Three - 3.', 'one two three'),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', '')
    ]
)
def test_clear_text_value(text: str, expected: str):
    result = clear(text)
    assert result == expected


@pytest.mark.parametrize(
    ['data'],
    [
        (pd.Series(data=[''], index=[0]), ),
        (pd.Series(data=['One', 'Two', 'Three'], index=range(3)), ),
        (pd.Series(data=['One - 1. Two - 2. Three - 3.'], index=[0]), ),
        (pd.Series(data=['1', '2', '3', '4', '5'], index=range(5)), )
    ]
)
def test_cleaning_data_type(data: pd.Series):
    result = isinstance(cleaning(data), pd.Series)
    assert result


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', ''),
        ('be is are', 'be be be'),
        ('cats dogs elephants mice', 'cat dog elephant mouse')
    ]
)
def test_lemmatize_text_value(text: str, expected: str):
    result = lemmatize(text)
    assert result == expected


@pytest.mark.parametrize(
    ['data'],
    [
        (pd.Series(data=[''], index=[0]), ),
        (pd.Series(data=['One', 'Two', 'Three'], index=range(3)), ),
        (pd.Series(data=['One - 1. Two - 2. Three - 3.'], index=[0]), ),
        (pd.Series(data=['1', '2', '3', '4', '5', ], index=range(5)), )
    ]
)
def test_lemmatization_data_type(data: pd.Series):
    result = isinstance(lemmatization(data), pd.Series)
    assert result
