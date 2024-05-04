from optuna.study import Study
from optuna.trial import FrozenTrial


class Verbose(object):
    """
    Класс для отображения прогресса подбора гиперпараметров с помощью Optuna.
    """

    def __init__(self, trials: int, name: str, n_job: int):
        """
        :param trials: общее количество испытаний;
        :param name: название модели;
        :param n_job: номер исследования;
        """

        self.trials = trials
        self.name = name
        self.n_job = n_job

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """
        Вызывается после каждого испытания;

        :param study: задача оптимизации - совокупность испытаний;
        :param trial: крайнее испытание;
        :return: None.
        """

        index = trial.number + 1
        state = trial.state.name
        complete = (trial
                    .datetime_complete
                    .strftime('%d-%m-%Y %H:%M:%S'))
        seconds = (trial.datetime_complete - trial.datetime_start).seconds
        minutes = seconds // 60
        seconds = seconds % 60
        value = round(trial.values[0], 4)
        best = round(study.best_value, 4)

        print(f'{self.name} [{self.n_job}]: '
              f'[{complete}] - [{minutes:02}:{seconds:02}] - '
              f'{state}: {index}/{self.trials} - {value:.4f} ({best:.4f}).')
