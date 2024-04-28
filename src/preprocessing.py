import os
import pandas as pd

from utils.data import preprocessing
from utils.explorer import explorer

from config.paths import PATH_RAW_DATA
from config.paths import PATH_PREPROCESSED_DATA


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

        data = preprocessing(
            games=games,
            genres=genres
        )

        # Сохранение предобработанных данных.
        data.to_csv(
            path_or_buf=fr'{PATH_PREPROCESSED_DATA}\{name}.csv',
            sep=',',
            index=False
        )


if __name__ == '__main__':
    main()
