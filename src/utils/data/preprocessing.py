import os

import numpy as np
import pandas as pd

from config.paths import PATH_PREPROCESSED_DATA
from config.paths import PATH_RAW_DATA


def indie(key: int, genres: pd.Series) -> bool:
    """
    Добавляет логическую переменную "indie";

    :param key: первичный ключ;
    :param genres: жанры видеоигр;
    :return: логическое значение или значение nan, если список пуст.
    """

    value = genres[key] if key in genres.index else False
    if isinstance(value, pd.Series):
        return True if 'Indie' in value.tolist() else False
    elif isinstance(value, str):
        return True if value == 'Indie' else False
    else:
        return False


def rating(key: int, scores: pd.DataFrame) -> float:
    """
    Рассчитывает рейтинг на основе поля "votes";

    :param key: первичный ключ;
    :param scores: голоса пользователей;
    :return: рейтинг.
    """

    values = scores.loc[key, :]
    if (total := values['amount'].sum()) >= 5:
        return (values['score'] * values['amount']).sum() / total
    else:
        return np.nan


def preprocessing(folder: str) -> None:
    """
    Обрабатывает данные:

    - удаляет явные дубликаты;
    - добавляет логическое поле "indie";
    - удаляет значения "indie" из поля "genres";
    - рассчитывает рейтинг на основе данных "scores";
    - добавляет поле "votes";
    - удаляет пустые записи;
    - изменяет типы данных;
    - удаляет неявные дубликаты;
    - удаляет записи с отрицательными значениями;

    :param folder: имя директории с данными;
    :return: None.
    """

    file_paths = {
        'games': f'{PATH_RAW_DATA}/{folder}/games.csv',
        'developers': f'{PATH_RAW_DATA}/{folder}/developers.csv',
        'genres': f'{PATH_RAW_DATA}/{folder}/genres.csv',
        'platforms': f'{PATH_RAW_DATA}/{folder}/platforms.csv',
        'scores': f'{PATH_RAW_DATA}/{folder}/scores.csv',
    }

    df = {file: pd.read_csv(path) for file, path in file_paths.items()}

    # Удаление явных дубликатов.
    for name in df:
        df[name] = df[name].drop_duplicates()

    # Добавление логического поля "indie".
    genres = df['genres'].set_index('id')['genre']
    df['games'].insert(
        loc=3,
        column='indie',
        value=(df['games']['id']
               .apply(indie, genres=genres)
               .values)
    )

    # Удаление значения "indie" из поля "genres"
    df['genres'] = df['genres'][df['genres']['genre'] != 'Indie']

    # Расчет рейтинга на основе данных "scores".
    scores = df['scores'].set_index('id')
    df['games'].loc[:, 'rating'] = (df['games']['id']
                                    .apply(rating, scores=scores)
                                    .astype('float32'))

    # Добавление поля "votes".
    df['games'].insert(
        loc=5,
        column='votes',
        value=(df['scores'][['id', 'amount']]
               .groupby('id')
               .sum()['amount'][df['games']['id']]
               .astype('int32')
               .values)
    )

    # Удаление пустых записей.
    columns = ['name', 'reviews', 'playing', 'backlogs', 'wishlists']
    mask = df['games'][columns].isna().any(axis=1)
    ids = df["games"][mask]['id']
    for name in df:
        df[name] = df[name][~df[name]['id'].isin(ids)].copy()

    df['games'].loc[df['games']['date'] == '6969-06-09', 'date'] = '2030-06-09'

    # Изменение типов данных
    df['games'] = df['games'].astype(
        {
            'votes': 'int32',
            'reviews': 'int32',
            'plays': 'int32',
            'playing': 'int32',
            'backlogs': 'int32',
            'wishlists': 'int32'
        }
    )

    # Удаление неявных дубликатов.
    mask = df['games'][["name", "date"]].duplicated()
    ids = df["games"][mask]['id']
    for name in df:
        df[name] = df[name][~df[name]['id'].isin(ids)]

    # Удаление записей с отрицательными значениями.
    columns = ['plays', 'playing', 'backlogs', 'wishlists']
    ids = df['games'][(df['games'][columns] < 0).any(axis=1)]['id']
    for name in df:
        df[name] = df[name][~df[name]['id'].isin(ids)]

    # Сохранение предобработанных данных.
    path = fr'{PATH_PREPROCESSED_DATA}\{folder}'
    if not os.path.exists(path):
        os.mkdir(path)
    for name, data in df.items():
        data.to_csv(
            path_or_buf=fr'{path}\{name}.csv',
            sep=',',
            index=False
        )
