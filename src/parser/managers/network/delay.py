import asyncio

from random import uniform


class DelayManager(object):
    """
    Менеджер задержки, задачами которого являются:

    - Задержка выполнения программы перед отправкой запроса;
    - Получение кода статуса отправленного запроса для масштабирования задержки;

    :var span: диапазон задержки;
    :var factor: масштаб задержки;
    :var type: тип текущей задержки;
    :var success: количество отправленных запросов с кодом 200;
    :var threshold: порог смены типа задержки.
    """

    def __init__(self):
        self.span: tuple[int, int] | None = None
        self.factor: int | None = None
        self.type: str = 'normal'
        self.success: int = 0
        self.threshold: int | None = None

    async def delay(self) -> None:
        """
        Задерживает выполнение программы перед отправкой запроса;

        :return: None.
        """

        if self.type == 'normal':
            span = self.span
        else:
            span = [s * self.factor for s in self.span]

        await asyncio.sleep(uniform(*span))

    async def code(self, code: int) -> None:
        """
        Получает код статуса отправленного запроса для масштабирования задержки;

        :param code: код статуса запроса;
        :return: None.
        """

        if code == 429:
            self.type = 'increased'
        elif code == 200 and self.type == 'increased':
            self.success += 1

            if self.success >= self.threshold:
                self.type = 'normal'
                self.success = 0

    def setting(self, span: tuple, factor: int, threshold: int) -> None:
        """
        Настраивает менеджер;

        :param span: диапазон задержки;
        :param factor: масштаб задержки;
        :param threshold: порог смены типа задержки;
        :return: None.
        """

        self.span = span
        self.factor = factor
        self.threshold = threshold
