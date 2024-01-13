import os

from config.paths import FILE_PREPROCESSED_PATH
from config.paths import MODELS_PATH
from ml.training import train
from utils.explorer import explorer


def main():
    """
    Тока входа тренировки моделей на предварительно обработанных данных;

    :return: None.
    """

    names = explorer(FILE_PREPROCESSED_PATH, '*.csv')
    os.system('cls')
    print('Список предобработанных файлов:', names, sep='\n', flush=True)

    if data := input('Выберите файл: '):
        models = []

        print(flush=True)
        names = explorer(MODELS_PATH, '*.py')
        print('Список файлов c моделями:', names, sep='\n', flush=True)

        if files := input('Выберите один или несколько файлов: '):
            for file in files.split():
                name = file.split('.')[0]

                modul = __import__(
                    name=f'ml.models.{name}',
                    globals=globals(),
                    locals=locals(),
                    fromlist=['title', 'model', 'params'],
                    level=0
                )

                models.append(
                    {
                        'name': name,
                        'title': modul.title,
                        'model': modul.model,
                        'params': modul.params,
                    }
                )

        train(file=data, models=models)


if __name__ == '__main__':
    main()
