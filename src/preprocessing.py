import os

from config.paths import PATH_RAW_DATA
from utils.data import preprocessing
from utils.explorer import explorer


def main():
    """
    Тока входа предварительной обработки данных;

    :return: None.
    """

    names = explorer(PATH_RAW_DATA, exclude=('checkpoints', ))
    os.system('cls')
    print('Список необработанных данных:', names, sep='\n', flush=True)

    if data := input('Выберите данные: '):
        preprocessing(data)


if __name__ == '__main__':
    main()
