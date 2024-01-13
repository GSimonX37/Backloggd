# Предварительная обработка данных

Точка входа для предварительной обработки данных находится в файле 
[preprocessing.py](../src/preprocessing.py):

```python
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
```

На данном этапе:
- удаляются явные дубликаты;
- удаляются пустые записи;
- удаляются записи с отрицательными значениями в полях: 
    - **reviews**;
    - **plays**;
    - **playing**;
    - **backlogs**;
    - **wishlists**.
- добавляется логическое поле **indie**;
- удаляются значения **indie** из поля **genres**;
- рассчитывается рейтинг на основе поля **votes**;
- удаляется поле **votes**;
- удаляются неявные дубликаты;
- изменяются типы данных.

Чтобы начать процесс предварительной обработки данных, 
необходимо запустить данный файл. Программа отобразит содержимое директории 
[raw](../data/raw), где хранятся файлы, 
сформированные на этапе сбора данных (см. [сбор данных](parsing.md)).

![file](../resources/preprocessing/file.jpg)

После предварительной обработки данных, 
в директории [processed](../data/processed) появится файл с данными в формате `.csv`. 
Название файла будет совпадать с названием файла в каталоге 
[raw](../data/raw).

>Обратите внимание, файлы из директории [raw](../data/raw) не удаляются.

[К описанию проекта](../README.md)
