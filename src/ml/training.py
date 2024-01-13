import ast
import json
import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from config.ml import FIT_CV_SPLITTING_STRATEGY
from config.ml import FIT_CV_VERBOSE
from config.ml import LEARNING_CURVE_SPLITTING_STRATEGY_N_SPLITS
from config.ml import LEARNING_CURVE_SPLITTING_STRATEGY_TEST_SIZE
from config.ml import LEARNING_CURVE_SPLITTING_STRATEGY_TRAIN_SIZES
from config.ml import LEARNING_CURVE_VERBOSE
from config.ml import N_JOBS
from config.ml import RANDOM_STATE
from config.ml import TEST_SIZE
from config.paths import FILE_PREPROCESSED_PATH
from config.paths import TRAIN_MODELS_REPORT_PATH
from config.paths import TRAINED_MODELS_PATH
from utils.ml.features import generate
from utils.ml.plot.balance import balance
from utils.ml.plot.calibration import calibration
from utils.ml.plot.metrics import metrics
from utils.ml.plot.scalability import scalability
from utils.ml.plot.words import words
from utils.ml.preprocessing import cleaning
from utils.ml.preprocessing import is_ascii
from utils.ml.preprocessing import lemmatization
from utils.ml.preprocessing import stop_words
from utils.ml.remove import remove


def train(file: str, models: list) -> None:
    """
    Обучает модели;

    :param file: имя предобработанного файла в формате csv;
    :param models: список с параметрами тренируемых моделей;
    :return: None.
    """

    df = pd.read_csv(fr'{FILE_PREPROCESSED_PATH}\{file}')

    # Преобразование поля "genres" к типу list.
    df['genres'] = df['genres'].apply(ast.literal_eval)

    # Отбор видеоигр, в описаниях которых присутствуют только ascii символы.
    data = df[df.apply(is_ascii, axis=1)]

    # Отбор данных.
    data = (data
            .loc[(data['description'].notna()) &
                 (data['genres'].map(bool)), ['description', 'genres']]
            .reset_index(drop=True))
    data = data[data['description'].str.len() > 50]

    data['genres'] = data['genres'].apply(remove)
    data = data[data['genres'].map(bool)].reset_index(drop=True)

    x = data[['description']]
    y = data['genres']

    # Создание препроцессора.
    generator = FunctionTransformer(generate)
    cleaner = ColumnTransformer(
        transformers=[
            ('preprocessor', FunctionTransformer(cleaning), [0]),
        ],
        remainder='passthrough'
    )
    lemmatizer = ColumnTransformer(
        transformers=[
            ('preprocessor', FunctionTransformer(lemmatization), [0]),
        ],
        remainder='passthrough'
    )
    preprocessor = Pipeline(
        steps=[
            ('generator', generator),
            ('cleaner', cleaner),
            ('lemmatizer', lemmatizer)
        ]
    )

    x = pd.DataFrame(
        data=preprocessor.fit_transform(x),
        columns=(preprocessor
                 .named_steps['cleaner']
                 .feature_names_in_)
    )

    # Кодирование меток.
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(y)
    labels = pd.Series(label_encoder.classes_)

    y = pd.DataFrame(label_encoder.transform(y))

    # Разделение на выборки.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=True,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    for model in models:
        name, title, model, params = (
            model['name'],
            model['title'],
            model['model'],
            model['params'],
        )

        path = fr'{TRAIN_MODELS_REPORT_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(fr'{path}\images'):
            os.mkdir(fr'{path}\images')

        # Сохранение отчета об частоте встречающихся слов в текстах.
        words(
            data=pd.concat(
                objs=[x[['description']], y],
                axis=1
            ),
            labels=labels,
            stop_words=stop_words,
            path=fr'{path}\images'
        )

        # Сохранение отчета о балансе классов в выборках.
        balance(
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

        # Обучение модели.
        clf = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=f1_weighted,
            cv=FIT_CV_SPLITTING_STRATEGY,
            verbose=FIT_CV_VERBOSE,
            refit=True
        )
        clf.fit(x_train, y_train)

        # Сохранение результатов кросс-валидации.
        (pd.DataFrame(data=clf.cv_results_)
         .sort_values('rank_test_score')
         .round(5)
         .to_csv(
            path_or_buf=rf'{path}\cv_results.csv',
            sep=',',
            index=False
        ))

        # Сохранение лучших гиперпараметров.
        with open(rf'{path}\best_params.json', 'w') as f:
            f.write(json.dumps(clf.best_params_, sort_keys=True, indent=4))

        # Оценка масштабируемости.
        (train_sizes,
         train_scores, test_scores,
         fit_times, score_times) = learning_curve(
            estimator=clf.best_estimator_,
            X=x_train,
            y=y_train,
            train_sizes=LEARNING_CURVE_SPLITTING_STRATEGY_TRAIN_SIZES,
            cv=ShuffleSplit(
                n_splits=LEARNING_CURVE_SPLITTING_STRATEGY_N_SPLITS,
                test_size=LEARNING_CURVE_SPLITTING_STRATEGY_TEST_SIZE,
                random_state=RANDOM_STATE
            ),
            n_jobs=N_JOBS,
            scoring=f1_weighted,
            return_times=True,
            verbose=LEARNING_CURVE_VERBOSE
        )

        x_train_size = x_train.shape[0]
        x_train_size *= (1 - LEARNING_CURVE_SPLITTING_STRATEGY_TEST_SIZE)
        scalability(
            train_sizes=pd.Series((train_sizes / x_train_size * 100).round(1)),
            train_scores=pd.DataFrame(train_scores),
            test_scores=pd.DataFrame(test_scores),
            fit_times=pd.DataFrame(fit_times),
            score_times=pd.DataFrame(score_times),
            title=f'Масштабируемость {title}',
            path=fr'{path}\images'
        )

        # Проверка на тестовой выборке
        predict = clf.predict(x_test)
        predict_proba = clf.predict_proba(x_test)

        f1 = f1_score(
            y_true=y_test,
            y_pred=predict,
            average='weighted'
        )

        metrics(
            y_test=y_test,
            y_predict=pd.DataFrame(predict),
            y_train=y_train,
            title=f'Результаты обучения {title} '
                  f'(F1-weighted: {f1:.4f}) на тестовой выборке',
            labels=labels,
            path=fr'{path}\images'
        )

        # Оценка калиброванности.
        calibration(
            y_true=y_test,
            y_proba=[pd.DataFrame(x) for x in predict_proba],
            labels=labels,
            title=f'График калиброванности {title}',
            path=fr'{path}\images'
        )

        path = fr'{TRAINED_MODELS_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        # Сохранение меток.
        with open(fr'{path}\labels.json', 'w') as file:
            file.write(json.dumps(labels.tolist()))

        model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', clf.best_estimator_)]
        )

        # Сохранение модели в joblib-файл.
        joblib.dump(
            value=model,
            filename=rf'{path}\{name}.joblib'
        )
