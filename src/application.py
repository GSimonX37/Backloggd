import os

import uvicorn

from app.application import model
from config.paths import PATH_TRAINED_MODELS
from utils.explorer import explorer


def main() -> None:
    """
    Тока входа запуска приложения;

    :return: None.
    """

    names = explorer(PATH_TRAINED_MODELS)
    os.system('cls')
    print('Список моделей:', names, sep='\n', flush=True)

    if directory := input('Выберите модель: '):
        file = PATH_TRAINED_MODELS + rf'\{directory}\{directory}.joblib'
        labels = PATH_TRAINED_MODELS + rf'\{directory}\labels.json'

        model.load(file, labels)

        uvicorn.run(
            app="app.application:app",
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )


if __name__ == "__main__":
    main()
