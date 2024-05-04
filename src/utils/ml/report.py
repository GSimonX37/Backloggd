from optuna.study import Study
from optuna.trial import FrozenTrial


class Report(object):
    """
    Класс для систематизации результатов исследования.
    """

    def __init__(self, n_job: int):
        """
        :param n_job: номер исследования;
        """

        self.n_job: int = n_job
        self.data: list[list] = []

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """
        Вызывается после каждого испытания;

        :param study: задача оптимизации - совокупность испытаний;
        :param trial: крайнее испытание;
        :return: None.
        """

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
        best = round(study.best_value, 4)

        self.data.append([
            self.n_job,
            index,
            state,
            start,
            complete
        ])

        self.data[-1] += [*params] + [value, best]
