import json

import joblib
import pandas as pd


class Model(object):
    """
    Модель, предсказывающая вероятности принадлежности видеоигры
    к игровым жанрам;

    :var model: модель;
    :var labels: метки классов.
    """

    def __init__(self):
        self.model = None
        self.labels: dict | None = None

    def load(self, model: str, labels: str) -> None:
        """
        Загружает модель и метки классов;

        :param model: полное имя файла модели в формате .joblib;
        :param labels: полное имя файла меток классов в формате .json.
        :return: None.
        """

        self.model = joblib.load(model)

        with open(labels) as f:
            self.labels = json.loads(f.read())

    def result(self, description: pd.Series, threshold: float) -> dict:
        """
        Предсказывает вероятности принадлежности видеоигры к игровым жанрам;

        :param description: описание видеоигры на английском языке;
        :param threshold: порог принятия решения о принадлежности объекта
                          к положительному классу;
        :return: вероятности, с которыми видеоигра принадлежит к игровым жанрам.
        """

        predict_proba = self.model.predict_proba(description)
        predict_proba = [proba[0, 1] for proba in predict_proba]

        genres = {}
        for label, proba in zip(self.labels.values(), predict_proba):
            if proba >= threshold:
                genres[label] = proba

        return genres
