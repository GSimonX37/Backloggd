import pandas as pd

from .preparation import spaces
from .preparation import length
from .preparation import letters
from .lemmatization import lemmatize


def insert(key: int, values: pd.Series) -> list:
    if key in values.index:
        value = values[key]
        return [value] if isinstance(value, str) else value.to_list()
    else:
        return []


def preprocessing(games: pd.DataFrame, genres: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    genres = genres.copy()

    # Удаление явных дубликатов.
    games = games.drop_duplicates()
    genres = genres.drop_duplicates()

    # Удаление значения "indie" из поля "genres"
    genres = genres[genres['genre'] != 'Indie']

    # Объединение данных.
    values = genres['genre'].value_counts().index[:20]
    genres = (genres
              .loc[genres['genre'].isin(values), :]
              .set_index('id'))['genre']

    games.insert(
        loc=games.shape[1],
        column='genres',
        value=games['id'].apply(insert, values=genres)
    )
    data = games[['description', 'genres']]

    # Удаление данных без целевой переменной.
    data = data.loc[data['genres'].map(bool), :]

    data = data.dropna()

    # Предварительная подготовка текста
    data['description'] = data['description'].apply(letters)
    data['description'] = data['description'].apply(length, size=2)
    data['description'] = data['description'].apply(spaces)

    # Лемматизация текста.
    data['description'] = data['description'].apply(lemmatize)

    # Удаление записей, с длиной текста меньше 50 символов.
    data = data[data['description'].str.len() >= 50]

    # Приведение текста к нижнему регистру.
    data['description'] = data['description'].str.lower()

    return data
