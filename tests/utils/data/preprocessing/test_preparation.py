import pytest

from src.utils.data.preprocessing.preparation import spaces
from src.utils.data.preprocessing.preparation import length
from src.utils.data.preprocessing.preparation import letters


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', ''),
        ('One. Two. Three. Four. Five.', 'One  Two  Three  Four  Five '),
        ('One - 1. Two - 2. Three - 3.', 'One      Two      Three     '),
        ('1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', '                             ')
    ]
)
def test_letters(text: str, expected: str):
    result = letters(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', ''),
        (' ', ' '),
        ('   ', ' '),
        ('\n\n\n', ' '),
        ('string', 'string'),
        ('string string', 'string string'),
        (' string ', ' string '),
        ('string\nstring\nstring\nstring', 'string string string string')
    ]
)
def test_spaces(text: str, expected: str):
    result = spaces(text)
    assert result == expected


@pytest.mark.parametrize(
    ['text', 'size', 'expected'],
    [
        ('', 2, ''),
        (' ', 2, ' '),
        ('Example of a small sentence', 2, 'Example     small sentence'),
    ]
)
def test_length(text: str, size: int, expected: str):
    result = length(text, size)
    assert result == expected
