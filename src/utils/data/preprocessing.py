import numpy as np
import pandas as pd

from config.paths import FILE_PREPROCESSED_PATH
from config.paths import FILE_RAW_PATH


def indie(genres: list) -> bool | float:
    """
    Добавляет логическую переменную "indie";

    :param genres: список жанров;
    :return: логическое значение или значение nan, если список пуст.
    """

    if genres:
        return True if 'Indie' in genres else False
    else:
        return np.nan


def remove(genres: list) -> list:
    """
    Удаляет значения "indie" из поля "genres";

    :param genres: список жанров;
    :return: список жанров без значения "Indie".
    """

    if 'Indie' in genres:
        genres.remove('Indie')
        return genres
    else:
        return genres


def rating(votes: list) -> float:
    """
    Рассчитывает рейтинг на основе поля "votes";

    :param votes: поле "votes";
    :return: рейтинг.
    """

    votes = np.array(votes)
    ratings = np.linspace(0.5, 5, 10)

    if sum(votes):
        return (ratings * votes).sum() / votes[votes > 0].sum()
    else:
        return np.nan


def preprocessing(file: str) -> None:
    """
    Обрабатывает данные:

    - удаляет явные дубликаты;
    - удаляет пустые записи;
    - удаляет записи с отрицательными значениями;
    - Добавляет логическое поле "indie";
    - удаляет значения "indie" из поля "genres";
    - рассчитывает рейтинг на основе поля "votes";
    - удаляет поле "votes";
    - удаляет неявные дубликаты;
    - изменяет типы данных;

    :param file: полное имя файла с очищенными данными в формате csv;
    :return: None.
    """

    name = fr'{FILE_RAW_PATH}\{file}'
    df = pd.read_csv(name, delimiter=';')

    # Удаление явных дубликатов.
    df = df.drop_duplicates()

    # Удаление пустых записей.
    columns = ['name',
               'date',
               'reviews',
               'plays',
               'playing',
               'backlogs',
               'wishlists']

    df = df[df[columns].notna().all(axis=1)]

    # Удаление записей, с отрицательными значениями.
    columns = ['reviews',
               'plays',
               'playing',
               'backlogs',
               'wishlists']

    df = df[(df[columns] >= 0).all(axis=1)]

    # Преобразование литералов списков к типу данных list.
    df["developers"] = df["developers"].apply(eval)
    df["platforms"] = df["platforms"].apply(eval)
    df["genres"] = df["genres"].apply(eval)
    df["votes"] = df["votes"].apply(lambda x: [*map(int, eval(x))])

    # Добавление логической переменной "indie".
    df.insert(
        loc=3,
        column='indie',
        value=df['genres'].apply(indie)
    )
    df['indie'] = df['indie'].fillna(False)
    # Удаление значения "indie" из поля "genres".
    df['genres'] = df['genres'].apply(remove)
    # Расчет рейтинга на основе поля "votes".
    df['rating'] = df['votes'].apply(rating)
    # Удаление поля "votes".
    df = df.drop(
        labels='votes',
        axis=1
    )

    # Удаление неявных дубликатов
    df = df[~df[["name", "date"]].duplicated()]

    # Изменение типов данных
    df = df.astype({
        "plays": "int32",
        "playing": "int32",
        "backlogs": "int32",
        "wishlists": "int32",
        "reviews": "int32"
    })

    df.to_csv(
        path_or_buf=fr'{FILE_PREPROCESSED_PATH}\{file}',
        sep=',',
        index=False
    )
