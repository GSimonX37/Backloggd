import os

import pandas as pd

from config.paths import PATH_PREPROCESSED_DATA
from config.paths import PATH_RAW_DATA
from utils.data import prepare
from utils.data import preprocess
from utils.explorer import explorer


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

        # Подготовка к предварительно обработке данных.
        data = prepare(games, genres)

        # Предварительная обработка данных.
        data['description'] = preprocess(data['description'])

        # Сохранение предобработанных данных.
        data.to_csv(
            path_or_buf=fr'{PATH_PREPROCESSED_DATA}\{name}.csv',
            sep=',',
            index=False
        )


if __name__ == '__main__':
    main()
