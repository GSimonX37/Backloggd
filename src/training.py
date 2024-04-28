import os
import pandas as pd
from config.paths import PATH_MODELS
from config.paths import PATH_PREPROCESSED_DATA
from ml.training import train
from utils.explorer import explorer


def main():
    """
    Тока входа тренировки моделей на предварительно обработанных данных;

    :return: None.
    """

    names = explorer(PATH_PREPROCESSED_DATA, ext='*.csv')
    os.system('cls')
    print('Список предварительно обработанных данных:', names,
          sep='\n',
          flush=True)

    if name := input('Выберите данные: '):
        data = pd.read_csv(f'{PATH_PREPROCESSED_DATA}/{name}')

        students = {}

        print(flush=True)
        names = explorer(path=PATH_MODELS,
                         ext='*.py',
                         exclude=('__init__.py', 'student.py'))
        print('Список файлов c моделями:', names, sep='\n', flush=True)

        if files := input('Выберите один или несколько файлов: '):
            for file in files.split():
                name = file.split('.')[0]

                modul = __import__(
                    name=f'ml.models.{name}',
                    globals=globals(),
                    locals=locals(),
                    fromlist=['student'],
                    level=0
                )

                students[name] = getattr(modul, 'student')

        train(
            students=students,
            data=data,
            n_trials=2
        )


if __name__ == '__main__':
    main()
