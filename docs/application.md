# Запуск приложения

Точка входа запуска приложения находится в файле 
[application.py](../src/application.py):

```python
import os

import uvicorn

from app.application import model
from config.paths import TRAINED_MODELS_PATH
from utils.explorer import explorer


def main() -> None:
    """
    Тока входа запуска приложения;

    :return: None.
    """

    names = explorer(TRAINED_MODELS_PATH)
    os.system('cls')
    print('Список моделей:', names, sep='\n', flush=True)

    if directory := input('Выберите модель: '):
        file = TRAINED_MODELS_PATH + rf'\{directory}\{directory}.joblib'
        labels = TRAINED_MODELS_PATH + rf'\{directory}\labels.json'

        model.load(file, labels)

        uvicorn.run(
            app="app.application:app",
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )


if __name__ == "__main__":
    main()
```

Чтобы приложение начало работу, необходимо запустить данный файл. 
Программа отобразит содержимое директории [models](../models), 
где хранятся модели, обученные на этапе тренировки моделей 
(см. [Тренировка моделей](training.md)).

![file](../resources/application/models.jpg)

После выбора модели, приложение начнет свою работу.

![file](../resources/application/run.jpg)

Запрос к приложению осуществляется посредством **post-запроса**, 
в теле которого должно быть указано:
1. description - описание видеоигры на английском языке;
2. threshold - порог принятия решения о принадлежности видеоигры к жанру.

Пример программы, осуществляющей запрос к приложению и получающей ответ от него:

```python
import requests

data = {
    'description': '',
    'thresholds': 0.5
}
r = requests.post('http://127.0.0.1:8000/', json=data)

print(r.content.decode('utf-8'))
```

Тело ответа на запрос содержит:
1. description - описание видеоигры на английском языке.
2. threshold - порог принятия решения о принадлежности видеоигры к жанру.
3. genres - предсказанные жанры с вероятностями к которым принадлежит видеоигра, 
с переданным в теле запроса описанием.

[К описанию проекта](../README.md)