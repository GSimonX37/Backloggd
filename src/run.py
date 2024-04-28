import sys

import uvicorn

from app.application import model


def main() -> None:
    """
    Тока входа запуска приложения в контейнере Docker;

    :return: None.
    """

    directory = sys.argv[1]

    file = rf'/models/{directory}/{directory}.joblib'
    labels = rf'/models/{directory}/labels.json'

    model.load(file, labels)

    uvicorn.run(
        app="app.application:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
