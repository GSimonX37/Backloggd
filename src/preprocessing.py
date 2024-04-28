import os
import pandas as pd

from utils.data import preprocessing
from utils.explorer import explorer

from config.paths import PATH_RAW_DATA
from config.paths import PATH_PREPROCESSED_DATA


def insert(key: int, values: pd.Series) -> list:
    if key in values.index:
        value = values[key]
        return [value] if isinstance(value, str) else value.to_list()
    else:
        return []


def main():
    """
    Тока входа предварительной обработки данных;

    :return: None.
    """

    names = explorer(PATH_RAW_DATA, exclude=('checkpoints', ))
    os.system('cls')
    print('Список необработанных данных:', names, sep='\n', flush=True)

    if name := input('Выберите данные: '):
        games = pd.read_csv(f'{PATH_RAW_DATA}/{name}/games.csv')
        genres = pd.read_csv(f'{PATH_RAW_DATA}/{name}/genres.csv')

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

        data['description'] = preprocessing(data['description'])

        # Удаление записей, с длиной текста меньше 50 символов.
        data = data[data['description'].str.len() >= 50]

        # Сохранение предобработанных данных.
        data.to_csv(
            path_or_buf=fr'{PATH_PREPROCESSED_DATA}\{name}.csv',
            sep=',',
            index=False
        )


if __name__ == '__main__':
    main()
