import time


class ProgressManager(object):
    """
    Менеджер прогресса, задачами которого являются:

    - учет текущего прогресса;
    - расчет оставшегося времени до завершения сбора данных;
    - расчет времени обработки 1 страницы (мин);
    - расчет текущей скорости обработки 1 страницы (стр/мин);
    - расчет пройденного времени с момента начала сора данных;

    :var start: время начала сбора данных;
    :var release: тип текущего релиза;
    :var progress: текущее количество страниц;
    :var finished: количество завершенных страниц;
    :var speed: текущая скорость обработки 1 страницы (стр/мин);
    :var interval: время обработки 1 страницы (мин);
    :var time: предыдущее измерение времени.
    """

    def __init__(self):
        self.start: float | None = None
        self.release: str | None = None
        self.progress: dict = {}
        self.finished: dict = {}
        self.speed: float | None = None
        self.interval: int | None = None
        self.time: int | None = None

    def setting(self, progress: list) -> None:
        """
        Настраивает менеджер;

        :param progress: текущий прогресс;
        :return: None.
        """

        self.progress = progress
        self.finished = [progress[0] - 1, progress[1]]

    def starting(self) -> None:
        """
        Начинает отсчет времени.

        :return: None.
        """
        self.start = time.time()
        self.time = time.time()

    def passed(self) -> float:
        """
        Вычисляет пройденное время с момента начала сора данных;

        :return: пройденное времени с момента начала сора данных.
        """

        return (time.time() - self.start) if self.start else 0

    def timer(self) -> None:
        """
        Вычисляет время обработки 1 страницы (мин.);

        :return: None.
        """

        if self.time:
            self.interval = round(time.time() - self.time, 2)
            self.time = time.time()
        else:
            self.time = time.time()
            self.interval = None

    def speeder(self) -> None:
        """
        Вычисляет скорость обработки 1 страницы (стр./мин.);

        :return: None.
        """

        if self.interval:
            self.speed = round(60 / self.interval, 2)
        else:
            self.speed = None

    async def next(self) -> None:
        """
        Увеличивает текущий прогресс;

        :return: None.
        """

        self.timer()
        self.speeder()
        self.finished[0] += 1
        self.progress[0] += 1

    def json(self) -> dict:
        """
        Возвращает текущие параметры;

        :return: Текущие параметры.
        """

        return {'progress': self.progress[0]}
