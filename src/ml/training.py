import ast
import csv
import json
import os

import joblib
import nltk
import optuna
import optuna.logging
import pandas as pd

from nltk.corpus import stopwords
from optuna.samplers import TPESampler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from config.ml import LEARNING_CURVE_N_SPLITS
from config.ml import LEARNING_CURVE_TEST_SIZE
from config.ml import LEARNING_CURVE_TRAIN_SIZES
from config.ml import N_JOBS
from config.ml import RANDOM_STATE
from config.ml import TEST_SIZE
from config.paths import PATH_TRAIN_REPORT
from config.paths import PATH_TRAINED_MODELS
from ml.models.model import Model
from utils import plot
from utils.ml.verbose import Verbose


optuna.logging.set_verbosity(optuna.logging.WARNING)
nltk.download('stopwords')


def train(models: dict[str: Model],
          data: pd.DataFrame,
          n_trials: int = 10,
          n_jobs: int = 1) -> None:
    """
    Обучает модели;

    :param models: словарь моделей;
    :param data: набор данных;
    :param n_trials: количество испытаний;
    :param n_jobs: количество ядер процессора,
    задействованных в подборе гипперпараметров;
    :return: None.
    """

    # Разделение на признаки.
    x = data['description']
    y = data['genres']

    y = y.apply(ast.literal_eval)

    encoder = MultiLabelBinarizer()
    encoder.fit(y)
    y = pd.DataFrame(encoder.transform(y))
    labels = pd.Series(encoder.classes_)

    # Разделение на выборки.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=True,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    for name, model in models.items():
        params = 1
        for param in model.params.values():
            params *= len(param[1])
        print(f'{model.name}: {n_trials}/{params} ({n_trials / params:.2%}).')

        model.x, model.y = x_train, y_train

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler()
        )

        verbose = Verbose(n_trials, model.name)

        study.optimize(
            model,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[verbose]
        )

        path = fr'{PATH_TRAIN_REPORT}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(fr'{path}\images'):
            os.mkdir(fr'{path}\images')

        # Сохранение отчета об частоте встречающихся слов в текстах.
        plot.words(
            data=pd.concat(
                objs=[x, y],
                axis=1
            ),
            labels=labels,
            stop_words=stopwords.words('english'),
            path=fr'{path}\images'
        )

        # Сохранение отчета о балансе классов в выборках.
        plot.balance(
            train=y_train,
            test=y_test,
            labels=labels,
            path=fr'{path}\images'
        )

        # Сохранение лучших гиперпараметров.
        with open(rf'{path}\params.json', 'w') as f:
            f.write(json.dumps(
                obj=study.best_params,
                sort_keys=True,
                indent=4)
            )

        # Сохранение результатов всех испытаний.
        trials = [[
            'index',
            'state',
            'start',
            'complete'
        ]]
        trials += [model.params.keys()] + ['values']
        for trial in study.trials:
            index = trial.number + 1
            state = trial.state.name
            start = (trial
                     .datetime_start
                     .strftime('%d-%m-%Y %H:%M:%S'))
            complete = (trial
                        .datetime_complete
                        .strftime('%d-%m-%Y %H:%M:%S'))
            params = trial.params.values()
            params = [round(p, 4) if isinstance(p, (int, float)) else p
                      for p in params]
            value = round(trial.values[0], 4)

            trials.append([
                index,
                state,
                start,
                complete
            ])

            trials[-1] += [*params] + [value]

        with open(fr'{path}\trials.csv', 'w',
                  newline='',
                  encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(trials)

        pipeline = model.pipeline.set_params(**study.best_params)
        pipeline.fit(x_train, y_train)

        # Оценка масштабируемости.
        (train_sizes,
         train_scores, test_scores,
         fit_times, score_times) = learning_curve(
            estimator=model,
            X=x_train,
            y=y_train,
            train_sizes=LEARNING_CURVE_TRAIN_SIZES,
            cv=ShuffleSplit(
                n_splits=LEARNING_CURVE_N_SPLITS,
                test_size=LEARNING_CURVE_TEST_SIZE,
                random_state=RANDOM_STATE
            ),
            n_jobs=N_JOBS,
            scoring=model.scoring,
            return_times=True,
            verbose=0
        )

        x_train_size = x_train.shape[0]
        x_train_size *= (1 - LEARNING_CURVE_TEST_SIZE)
        plot.scalability(
            train_sizes=pd.Series((train_sizes / x_train_size * 100).round(1)),
            train_scores=pd.DataFrame(train_scores),
            test_scores=pd.DataFrame(test_scores),
            fit_times=pd.DataFrame(fit_times),
            score_times=pd.DataFrame(score_times),
            title=f'Масштабируемость модели {model.name}',
            path=fr'{path}\images'
        )

        predict = pipeline.predict(x_test)
        predict_proba = pipeline.predict_proba(x_test)

        metric = model.metric(y_test, predict)

        plot.metrics(
            y_test=y_test,
            y_predict=pd.DataFrame(predict),
            y_train=y_train,
            title=f'Результаты обучения модели {model.name} '
                  f'(F1-weighted: {metric:.4f}) на тестовой выборке',
            labels=labels,
            path=fr'{path}\images'
        )

        # Оценка калиброванности.
        plot.calibration(
            y_true=y_test,
            y_proba=[pd.DataFrame(x) for x in predict_proba],
            labels=labels,
            title=f'График калиброванности модели {model.name}',
            path=fr'{path}\images'
        )

        # Сравнение с простым классификатором.
        dummy = DummyClassifier(
            strategy='stratified',
            random_state=RANDOM_STATE
        )
        dummy.fit(x_train, y_train)

        predict = pd.DataFrame(dummy.predict(x_test))

        metric = model.metric(y_test, predict)

        plot.metrics(
            y_test=y_test,
            y_predict=pd.DataFrame(predict),
            y_train=y_train,
            title=f'Результаты обучения простой эмпирической модели '
                  f'(F1-weighted: {metric:.4f}) на тестовой выборке',
            labels=labels,
            name='dummy',
            path=fr'{path}\images'
        )

        path = fr'{PATH_TRAINED_MODELS}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        # Сохранение меток.
        with open(fr'{path}\labels.json', 'w') as file:
            file.write(json.dumps(
                obj={code: label for code, label in labels.items()},
                sort_keys=True,
                indent=4)
            )

        # Сохранение модели в joblib-файл.
        joblib.dump(
            value=pipeline,
            filename=rf'{path}\{name}.joblib'
        )
