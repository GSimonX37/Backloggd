import pytest

from src.utils.data.preprocessing.lemmatization import lemmatize


@pytest.mark.parametrize(
    ['text', 'expected'],
    [
        ('', ''),
        ('be is are', 'be be be'),
        ('cats dogs elephants mice', 'cat dog elephant mouse')
    ]
)
def test_lemmatize(text: str, expected: str):
    result = lemmatize(text)
    assert result == expected
