import pandas as pd
import numpy as np

from optuna import Trial
from sklearn.model_selection import cross_validate


class Student(object):
    """
    Класс для подбора гипперпараметров с помощью Optuna.
    """

    def __init__(self,
                 model,
                 params: dict,
                 scorer: callable,
                 scoring: callable,
                 cv,
                 n_jobs: int = 1):

        """
        :param model: pipeline модели;
        :param params: пространство гиперпараметров;
        :param scorer: функция оценки модели на тестовой выборке;
        :param scoring: функция оценки модели во время кросс валидации;
        :param cv: метод кросс валидации;
        :param n_jobs: количество ядер процессора,
        задействованных на кросс валидации;
        """

        self.model = model
        self.params: dict = params
        self.scorer: callable = scorer
        self.scoring: callable = scoring
        self.cv = cv
        self.n_jobs: int = n_jobs

        self.x: pd.Series | None = None
        self.y: pd.Series | None = None

    def __call__(self, trial: Trial) -> float:
        """
        Метод, используемый при обучении модели с помощью Optuna;

        :param trial: испытание – процесс оценки целевой функции;
        :return: оценка модели.
        """

        # Задание гиперпараметров.
        params = {}
        for name, (t, values) in self.params.items():
            if t == 'int':
                params[name] = trial.suggest_int(name, **values)
            elif t == 'float':
                params[name] = trial.suggest_float(name, **values)
            elif t == 'categorical':
                params[name] = trial.suggest_categorical(name, *values)

        # Инициализация модели.
        model = self.model.set_params(**params)

        results: np.ndarray = cross_validate(
                estimator=model,
                X=self.x,
                y=self.y,
                scoring=self.scoring,
                cv=self.cv,
                verbose=0,
                n_jobs=self.n_jobs
        )

        return results['test_score'].mean()