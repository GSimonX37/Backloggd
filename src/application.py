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
