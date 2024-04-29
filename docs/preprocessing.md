# Предварительная обработка данных

Предварительная обработка данных необходима, чтобы сделать "сырые" данные
пригодными для машинного обучения.
Точка входа для предварительной обработки данных находится в файле 
[preprocessing.py](../src/preprocessing.py):

```python
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
```

На данном этапе:
1. Предварительная подготовка:
    - удаляются явные дубликаты;
    - удаляются значения "indie" из поля "genres";
    - добавляются целевые переменные и отбираются необходимых данные;
    - удаляются пропуски.
2. Предварительная обработка:
    - очистка текстовых данных;
    - лемматизация текста.

Чтобы начать процесс предварительной обработки данных, 
необходимо запустить данный файл. Программа отобразит содержимое каталога 
[raw](../data/raw), где хранятся данные, 
сформированные на этапе сбора данных (см. [сбор данных](parsing.md)).

![file](../resources/preprocessing/file.jpg)

После предварительной обработки данных, 
в каталоге [processed](../data/processed) появятся данные в формате `.csv`. 
Название файла будет совпадать с названием каталога в каталоге 
[raw](../data/raw).

>Обратите внимание, файлы из директории [raw](../data/raw) не удаляются.

[К описанию проекта](../README.md)
