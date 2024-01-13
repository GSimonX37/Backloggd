import numpy as np
import pandas as pd
import pytest

from src.utils.ml.features import digits_number
from src.utils.ml.features import generate
from src.utils.ml.features import letters_number
from src.utils.ml.features import lower_words_number
from src.utils.ml.features import punctuation_mark_number
from src.utils.ml.features import title_words_number
from src.utils.ml.features import upper_words_number
from src.utils.ml.features import words_number


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 0),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 5),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 10)
    ]
)
def test_digits_number_value(text, expected):
    result = digits_number(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 19),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 19),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 0)
    ]
)
def test_letters_number_value(text, expected):
    result = letters_number(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 14),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 14),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 0)
    ]
)
def test_lower_words_number_value(text, expected):
    result = lower_words_number(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 5),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 10),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 10)
    ]
)
def test_punctuation_mark_number_value(text, expected):
    result = punctuation_mark_number(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'output'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 5),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 5),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 0)
    ]
)
def test_title_words_number_value(text, output):
    result = title_words_number(text)
    assert result == output


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 5),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 5),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 0)
    ]
)
def test_upper_words_number_value(text, expected):
    result = upper_words_number(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', 0),
        ('One. Two. Three. Four. Five.', 5),
        ('One - 1. Two - 2. Three - 3. Four - 5. Five - 5.', 15),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 10)
    ]
)
def test_words_number_value(text, expected):
    result = words_number(text)
    assert result == expected


@pytest.mark.parametrize(
    ['data'],
    [
        (pd.DataFrame(data=[''], index=[0]), ),
        (pd.DataFrame(data=['One', 'Two', 'Three'], index=range(3)), ),
        (pd.DataFrame(data=['One - 1. Two - 2. Three - 3.'], index=[0]), ),
        (pd.DataFrame(data=['1', '2', '3', '4', '5'], index=range(5)), )
    ]
)
def test_features_data_type(data):
    result = isinstance(generate(data), pd.DataFrame)
    assert result


@pytest.mark.parametrize(
    ['data'],
    [
        (pd.DataFrame(data=[''], index=[0]),),
        (pd.DataFrame(data=['One', 'Two', 'Three'], index=range(3)),),
        (pd.DataFrame(data=['One - 1. Two - 2. Three - 3'], index=[0]), ),
        (pd.DataFrame(data=['1', '2', '3', '4', '5'], index=range(5)), )
    ]
)
def test_features_data_types(data):
    object_types = isinstance(data.dtypes.values[0], np.dtypes.ObjectDType)

    integer_types = data.dtypes.values[1:]
    integer_types = [isinstance(t, np.dtypes.Int64DType) for t in integer_types]
    result = object_types and all(integer_types)
    assert result


@pytest.mark.parametrize(
    ['data', 'expected'],
    [
        (pd.DataFrame(data=[''], index=[0]), (1, 17)),
        (pd.DataFrame(data=['One', 'Two', 'Three'], index=range(3)), (3, 17)),
        (pd.DataFrame(data=['One - 1. Two - 2.'], index=[0]), (1, 17)),
        (pd.DataFrame(data=['1', '2', '3', '4', '5'], index=range(5)), (5, 17))
    ]
)
def test_features_shape(data, expected):
    result = generate(data).shape
    assert result == expected
