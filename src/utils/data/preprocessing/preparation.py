import pandas as pd


def insert(key: int, values: pd.Series) -> list:
    """
    Формирует список по ключу "key" из значений "values";

    :param key: ключ;
    :param values: значения;
    :return: список значений;
    """

    if key in values.index:
        value = values[key]
        return [value] if isinstance(value, str) else value.to_list()
    else:
        return []


def prepare(games, genres) -> pd.DataFrame:
    """
    Подготавливает данные для предварительной обработки;

    :param games: данные о видеоиграх;
    :param genres: данные о жанрах видеоигр;
    :return: подготовленный набор данных.
    """

    games = games.copy()
    genres = genres.copy()

    # Удаление явных дубликатов.
    games = games.drop_duplicates()
    genres = genres.drop_duplicates()

    # Удаление значения "indie" из поля "genres".
    genres = genres[genres['genre'] != 'Indie']

    # Добавление целевой переменной.
    values = genres['genre'].value_counts().index[:20]
    genres = (genres
              .loc[genres['genre'].isin(values), :]
              .set_index('id'))['genre']

    games.insert(
        loc=games.shape[1],
        column='genres',
        value=games['id'].apply(insert, values=genres)
    )

    # Отбор необходимых данных.
    data = games[['description', 'genres']]

    # Удаление пропусков.
    data = data.loc[data['genres'].map(bool), :]
    data = data.dropna()

    return data
