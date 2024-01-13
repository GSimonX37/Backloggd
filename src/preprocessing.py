import os

from config.paths import FILE_RAW_PATH
from utils.data.preprocessing import preprocessing
from utils.explorer import explorer


def main():
    """
    Тока входа предварительной обработки данных;

    :return: None.
    """

    names = explorer(FILE_RAW_PATH, '*.csv')
    os.system('cls')
    print('Список необработанных файлов:', names, sep='\n', flush=True)

    if data := input('Выберите файл: '):
        preprocessing(data)


if __name__ == '__main__':
    main()
