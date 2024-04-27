from optuna.study import Study
from optuna.trial import FrozenTrial


class Verbose(object):
    def __init__(self, trials: int):
        self.trials = trials

    def __call__(self, study: Study, trial: FrozenTrial):
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

        print(f'[{complete}] - [{minutes:02}:{seconds:02}] - '
              f'{state}: {index}/{self.trials} - {value:.4f} ({best:.4f}).')
