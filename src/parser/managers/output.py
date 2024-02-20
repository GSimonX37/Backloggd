import asyncio
import os


class ProgressBar:
    """
    Шкала отображения прогресса;

    :var current: текущее значение прогресса;
    :var maximum: максимальное значение прогресса;
    :var time: расчетное оставшееся время до завершения сбора данных.
    """

    def __init__(self):
        self.current: int | None = None
        self.maximum: int | None = None
        self.time: int | None = None

    def step(self, current: int, maximum: int, speed: float) -> None:
        """
        Изменяет состояние прогресса;

        :param current: текущее значение прогресса;
        :param maximum: максимальное значение прогресса;
        :param speed: текущая скорость.
        :return: None.
        """

        self.current = current
        self.maximum = maximum
        self.time = ((maximum - current) // speed) if speed else None

    def __str__(self) -> str:
        h = int(self.time // 60) if self.time else 0
        m = int(self.time % 60) if self.time else 0
        percents = 100 / self.maximum * self.current
        marks = int(round(percents, 2) / 2)

        return (f'|{chr(0x2588) * marks:50}| - {percents:6.2f}% '
                f'({self.current} из {self.maximum}) - {h:02} час. {m:02} мин.')


class OutputManager:
    """
    Менеджер вывода, задачами которого являются:

    - отображение текущего состояния сбора данных;

    :var passed: пройденное времени с момента начала сора данных;
    :var timeout: задержка между выводами текущего состояния;
    :var states: состояния менеджеров для отображения на экране;
    :var stopped: флаг остановки вывода данных;
    :var release: прогресс сбора данных текущего релиза;
    :var total: прогресс сбора всех данных.
    """

    def __init__(self):
        self.passed: float | None = None
        self.timeout: int | None = None
        self.states: dict = {}
        self.stopped: bool = False
        self.release: ProgressBar = ProgressBar()
        self.total: ProgressBar = ProgressBar()

    async def file(self, file: str, size: int, records: int) -> None:
        """
        Получает данные и формирует состояние файлового менеджера;

        :param file: имя файла с данными;
        :param size: размер файла с данными;
        :param records: количество собранных данных;
        :return: None
        """

        self.states['file'] = (
            f'Имя файла: {file:>24}.\n'
            f'Размер файла: {size / 2 ** 10:18.2f} KB.\n'
            f'Количество записей: {records:15}.'
        )

    async def network(self, statuses: dict, traffic: int, span: tuple) -> None:
        """
        Получает данные и формирует состояние сетевого менеджера;

        :param statuses: статусы отправленных запросов;
        :param traffic: размер входящего трафика;
        :param span: диапазон задержки;
        :return: None.
        """

        failed = ''
        for (status, count) in statuses["failed"].items():
            failed += f'{"-":>5} {status:<6} {count:22};\n'

        total = statuses["successful"]
        total += sum(statuses["failed"].values())
        spans = f'{span[0]} - {span[1]}'

        self.states['network'] = (
            f'Входящий трафик: {traffic / 2 ** 10:15.2f} KB.\n'
            f'Задержка: {spans:>21} сек.\n'
            f'Коды статусов отправленных запросов:\n'
            f'{"Успешно":10} {statuses["successful"]:24};\n'
            f'{"Неуспешно":10} {sum(statuses["failed"].values()):24};\n'
            f'{failed}'
            f'{"Всего":10} {total:24}.'
        )

    async def parsing(self, success: dict, failed: dict) -> None:
        """
        Получает данные и формирует состояние менеджера парсинга;

        :param success: успешно спарсенные данные;
        :param failed: неуспешно спарсенные данные;
        :return: None.
        """

        data = []
        fields = zip(success.items(), failed.values())
        for (field, success), failed in fields:
            total = failed + success
            percent = (success / total) if total else 0
            data += [f'{field:11} {success:6} из {total:6} - {percent:7.2%}']

        data = ';\n'.join(data)

        self.states['parsing'] = (
            f'Количество успешно обработанных данных:\n'
            f'{data}.'
        )

    async def progress(self,
                       passed: float,
                       finish: dict,
                       speed: float,
                       interval: int) -> None:
        """
        Получает данные и формирует состояние менеджера прогресса;

        :param passed: пройденное время с момента начала сора данных;
        :param finish: количество завершенных страниц;
        :param speed: текущая скорость обработки 1 страницы (стр./мин.);
        :param interval: время обработки 1 страницы (мин.);
        :return: None.
        """

        self.passed = passed

        releases = ''
        for r, (current, maximum) in finish.items():
            percent = current / maximum
            releases += f'{r:11} {current:6} из {maximum:6} - {percent:7.2%};\n'

        release = ''
        for (release, (current, maximum)) in finish.items():
            if current != maximum:
                release = release
                break

        self.release.step(
            current=finish[release][0],
            maximum=finish[release][1],
            speed=speed
        )

        current = sum([current for (current, maximum) in finish.values()])
        maximum = sum([maximum for (current, maximum) in finish.values()])
        percent = current / maximum
        releases += f'{"Всего":11} {current:6} из {maximum:6} - {percent:7.2%}.'

        self.total.step(
            current=current,
            maximum=maximum,
            speed=speed
        )

        speed = speed if speed else 0.0
        interval = interval if interval else 0

        self.states['progress'] = (
            f'Скорость обработки: {speed:9.2f} стр./мин.\n'
            f'Время обработки: {interval:12} сек./стр.\n'
            f'{releases}\n\n'
            f'Релиз: {self.release}\n'
            f'Всего: {self.total}'
        )

    def setting(self, timeout: int) -> None:
        """
        Настраивает менеджер;

        :param timeout: задержка между выводами текущего состояния;
        :return: None.
        """

        self.timeout = timeout

    async def state(self, repeat: bool = False) -> None:
        """
        Отображает текущее состояние сбора данных;

        :param repeat: повтор вывода текущего состояния;
        :return: None.
        """

        while not self.stopped:
            s = self.passed
            h, m, s = int(s // 3600), int(s % 3600 // 60), int(s % 3600 % 60)

            states = '\n\n'.join(self.states.values())

            state = (
                f'Время выполнения: {h:02} час. {m:02} мин. {s:02} сек.\n'
                f'\n'
                f'{states}'
            )

            os.system('cls')

            print(state, flush=True)

            if not repeat:
                break

            await asyncio.sleep(self.timeout)

    def json(self) -> dict:
        """
        Возвращает текущие параметры:

        - timeout: задержка между выводами текущего состояния;

        :return: Текущие параметры.
        """

        return {'timeout': self.timeout}
