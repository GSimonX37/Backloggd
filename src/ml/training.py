import csv
import json
import os
import optuna
import joblib
import optuna.logging
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from config.ml import LEARNING_CURVE_N_SPLITS
from config.ml import LEARNING_CURVE_TEST_SIZE
from config.ml import LEARNING_CURVE_TRAIN_SIZES
from config.ml import N_JOBS
from config.ml import RANDOM_STATE
from config.ml import TEST_SIZE
from config.paths import PATH_PREPROCESSED_DATA
from config.paths import PATH_TRAIN_REPORT
from config.paths import PATH_TRAINED_MODELS
from utils import plot
from utils.ml.preprocessing import cleaning
from utils.ml.preprocessing import lemmatization
from utils.ml.preprocessing import stopwords

from ml.models.student import Student
from utils.ml.verbose import Verbose
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def insert(key: int, values: pd.Series) -> list:
    """
    Группирует значения по ключу в список;

    :param key: ключ, по которому группируются значения;
    :param values: значения;
    :return: список значений.
    """

    if key in values.index:
        value = values[key]
        return [value] if isinstance(value, str) else value.to_list()
    else:
        return []


def train(students: dict[str: Student],
          folder: str,
          n_trials: int = 10,
          n_jobs: int = 1) -> None:
    """
    Обучает модели;

    :param students: словарь моделей;
    :param folder: директория с предварительно обработанными данными;
    :param n_trials: количество испытаний;
    :param n_jobs: количество ядер процессора,
    задействованных в подборе гипперпараметров;
    :return: None.
    """

    file_paths = {
        'games': f'{PATH_PREPROCESSED_DATA}/{folder}/games.csv',
        'genres': f'{PATH_PREPROCESSED_DATA}/{folder}/genres.csv',
    }

    df = {file: pd.read_csv(path) for file, path in file_paths.items()}

    # Отбор описаний видеоигр.
    data = df['games'].loc[:, ['id', 'description']].copy()

    # Удаление записей без описания.
    data = (data
            .loc[data['description'].notna(), :]
            .reset_index(drop=True))

    # Отбор записей только с ascii-символами.
    data = (data
            .loc[data['description'].apply(lambda s: s.isascii()), :]
            .reset_index(drop=True))

    # Отбор 20 самых популярных жанров.
    genres = df['genres']['genre'].value_counts().index[:20]
    genres = (df['genres']
              .loc[df['genres']['genre'].isin(genres), :]
              .set_index('id'))['genre']

    # Объединение данных.
    data.insert(
            loc=data.shape[1],
            column='genres',
            value=data['id'].apply(insert, values=genres)
    )
    data = data.drop('id', axis=1)

    data = (data
            .loc[data['genres'].map(bool), :]
            .reset_index(drop=True))

    # Создание препроцессора.
    cleaner = FunctionTransformer(cleaning)
    lemmatizer = FunctionTransformer(lemmatization)
    preprocessor = Pipeline(
        steps=[
            ('cleaner', cleaner),
            ('lemmatizer', lemmatizer)
        ]
    )

    # Очистка и лемматизация текста
    data['description'] = preprocessor.fit_transform(data['description'])

    # Отбор записей, описание которых не состоит только из пробельных символов.
    data = (data
            .loc[~data['description'].apply(lambda s: s.isspace()), :]
            .reset_index(drop=True))

    # Разделение на признаки.
    x = data['description']
    y = data['genres']

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

    verbose = Verbose(n_trials)

    for name, student in students.items():
        params = 1
        for param in student.params.values():
            params *= len(param[1])
        print(f'{student.name}: {n_trials}/{params} ({n_trials / params:.2%}).')

        student.x, student.y = x_train, y_train

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler()
        )

        study.optimize(
            student,
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
            stop_words=stopwords,
            path=fr'{path}\images'
        )

        # Сохранение отчета о балансе классов в выборках.
        plot.balance(
            train=y_train,
            test=y_test,
            labels=labels,
            path=fr'{path}\images'
        )

        # Метрика оценки классификатора.
        f1_weighted = make_scorer(
            score_func=f1_score,
            average='weighted',
            zero_division=0.0
        )

        # Сохранение лучших гиперпараметров.
        with open(rf'{path}\params.json', 'w') as f:
            f.write(json.dumps(
                study.best_params,
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
        trials += [student.params.keys()] + ['values']
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

        model = student.model.set_params(**study.best_params)
        model.fit(x_train, y_train)

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
            scoring=f1_weighted,
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
            title=f'Масштабируемость модели {student.name}',
            path=fr'{path}\images'
        )

        predict = model.predict(x_test)
        predict_proba = model.predict_proba(x_test)

        f1 = f1_score(
            y_true=y_test,
            y_pred=predict,
            average='weighted'
        )

        plot.metrics(
            y_test=y_test,
            y_predict=pd.DataFrame(predict),
            y_train=y_train,
            title=f'Результаты обучения модели {student.name} '
                  f'(F1-weighted: {f1:.4f}) на тестовой выборке',
            labels=labels,
            path=fr'{path}\images'
        )

        # Оценка калиброванности.
        plot.calibration(
            y_true=y_test,
            y_proba=[pd.DataFrame(x) for x in predict_proba],
            labels=labels,
            title=f'График калиброванности модели {student.name}',
            path=fr'{path}\images'
        )

        # Сравнение с простым классификатором.
        dummy_clf = DummyClassifier(
                strategy='stratified',
                random_state=RANDOM_STATE
        )
        dummy_clf.fit(x_train, y_train)

        predict = pd.DataFrame(dummy_clf.predict(x_test))

        f1 = f1_score(
            y_true=y_test,
            y_pred=predict,
            average='weighted'
        )

        plot.metrics(
            y_test=y_test,
            y_predict=pd.DataFrame(predict),
            y_train=y_train,
            title=f'Результаты обучения простой эмпирической модели '
                  f'(F1-weighted: {f1:.4f}) на тестовой выборке',
            labels=labels,
            name='dummy',
            path=fr'{path}\images'
        )

        path = fr'{PATH_TRAINED_MODELS}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        # Сохранение меток.
        with open(fr'{path}\labels.json', 'w') as file:
            file.write(json.dumps(labels.tolist()))

        # Сохранение модели в joblib-файл.
        joblib.dump(
            value=model,
            filename=rf'{path}\{name}.joblib'
        )
