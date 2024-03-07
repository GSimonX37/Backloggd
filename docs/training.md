# Тренировка и оценка моделей

Точка входа тренировки моделей находится в файле 
[training.py](../src/training.py):

```python
import os

from config.paths import PATH_PREPROCESSED_DATA
from config.paths import PATH_MODELS
from ml.training import train
from utils.explorer import explorer


def main():
    """
    Тока входа тренировки моделей на предварительно обработанных данных;

    :return: None.
    """

    names = explorer(PATH_PREPROCESSED_DATA)
    os.system('cls')
    print('Список предварительно обработанных данных:', names,
          sep='\n',
          flush=True)

    if data := input('Выберите данные: '):
        models = []

        print(flush=True)
        names = explorer(PATH_MODELS, '*.py')
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

        train(folder=data, models=models)


if __name__ == '__main__':
    main()
```

## Тренировка моделей

Чтобы начать процесс тренировки моделей, необходимо запустить данный файл. 
Программа отобразит содержимое каталога [processed](../data/processed), 
где хранятся файлы, сформированные на этапе предварительной обработки данных 
(см. [Предварительная обработка данных](preprocessing.md)). Необходимо выбрать каталог 
с данными, на которых модель будет обучаться.

![file](../resources/training/file.jpg)

После выбора данных, на которых будет проводиться тренировка, 
необходимо выбрать одну или несколько моделей. 

![models](../resources/training/models.jpg)

Все модели должны располагаться в каталоге [models](../src/ml/models), 
с расширением `*.py` и иметь следующее содержимое:
1. title - заголовок, который будет использован при построении отчетов.
2. model - модель машинного обучения.
3. params - гиперпараметры модели.

```python
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


title = 'SGDClassifier'

vectorizer = TfidfVectorizer(
    analyzer='word'
)

standardizer = Pipeline(
    steps=[
        ('vectorizer', vectorizer)
    ]
)

estimator = SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    random_state=42
)

estimator = MultiOutputClassifier(
    estimator=estimator,
    n_jobs=4
)

model = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator)
    ]
)

params = {
    'standardizer__vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'standardizer__vectorizer__norm': [None, 'l1', 'l2'],
    'standardizer__vectorizer__max_features': np.arange(
        start=750_000,
        stop=1_000_001,
        step=250_000
    ).tolist(),
    'estimator__estimator__alpha': np.linspace(
        start=0.1,
        stop=0.5,
        num=5
    ).round(5).tolist(),
    'estimator__estimator__class_weight': [None, 'balanced'],
    'estimator__estimator__l1_ratio': np.linspace(
        start=0.0,
        stop=0.5,
        num=6
    ).round(5).tolist()
}
```

## Оценка моделей

После завершения обучения, в каталоге [models](../models) 
будет создан каталог с названием файла модели, 
указанного перед началом тренировки. В каталоге будут находиться файлы: 
- `labels.json` - метки классов, наблюдаемые в данных, 
во время тренировки модели.
- файл обученной модели с расширением `*.joblib`, 
имя которого будет совпадать с названием файла модели, 
указанного перед началом тренировки.

В каталоге [training](../reports/training) будет создан каталог 
с названием файла модели, указанного перед началом тренировки. 
В папке будут находиться: 
1. Файл `best_params.json` - гиперпараметры модели, 
при которых предсказательная способность модели была наилучшей.
2. Файл `cv_results.csv` - результаты кросс-валидации.
3. Каталог `images` - графические материалы.

В каталоге `images` будут содержаться следующие файлы:
- `words.png` - результаты частотного анализа;
- `balance.png` - баланс классов в тренировочной и тестовой выборах;
- `metrics.png` - предсказательная способность модели;
- `scalability.png` - масштабируемость модели;
- `calibration.png` - калиброванность модели;
- `dummy.png` - предсказательная способность простой модели.

Примеры графических материалов, сформированных по результатам тренировки модели:

![words](../resources/training/words.png)

![balance](../resources/training/balance.png)

![metrics](../resources/training/metrics.png)

![scalability](../resources/training/scalability.png)

![calibration](../resources/training/calibration.png)

![dummy](../resources/training/dummy.png)


[К описанию проекта](../README.md)