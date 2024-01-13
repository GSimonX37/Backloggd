import aiohttp

from config.parser.managers.network.network import HEADERS
from config.parser.managers.network.network import URL
from config.parser.spider import RELEASES
from parser.managers.network.delay import DelayManager


class NetworkManager:
    """
    Сетевой менеджер, задачами которого являются:

    - Создание клиентской сессии;
    - Проверка соединения с сервером перед началом сбора данных;
    - Отправление get-запроса по указанному адресу;
    - учет размера входящего трафика;
    - учет статусов отправленных запросов;
    - хранение и выдача адресов страниц для сбора данных;

    :var delay: менеджер задержки;
    :var headers: заголовки get-запросов;
    :var traffic: размер входящего трафика;
    :var releases: названия релизов в адресной строке;
    :var url: адрес сайта web-ресурса;
    :var session: клиентская сессия для отправления запросов;
    :var statuses: статусы отправленных запросов;
    """

    def __init__(self):
        self.delay: DelayManager = DelayManager()

        self.headers: dict = HEADERS
        self.traffic: int = 0
        self.releases: dict[str, str] = RELEASES
        self.url: str = URL
        self.session: aiohttp.ClientSession | None = None
        self.statuses: dict = {
            "successful": 0,
            "failed": {}
        }

    async def connect(self) -> int:
        """
        Создает экземпляр клиентской сессии ClientSession. Проверяет соединение
        с сервером перед началом сбора данных;

        :return: код статуса ответа на запрос.
        """

        self.session = aiohttp.ClientSession(
            headers=self.headers
        )

        async with self.session.get(self.url) as response:
            return response.status

    async def get(self, link: str, params: dict = None) -> dict:
        """
        Отправляет get-запрос по указанному адресу. Учитывает размер входящего
        трафика и статусы отправленных запросов;

        :param link: адрес, по которому будет отправлен запрос;
        :param params: параметры запроса;
        :return: код статуса запроса, текст тела запроса.
        """

        await self.delay.delay()

        async with self.session.get(link, params=params) as response:
            code = response.status

            await self.delay.code(code)

            if code == 200:
                self.statuses["successful"] += 1
            else:
                if code in self.statuses["failed"]:
                    self.statuses["failed"][code] += 1
                else:
                    self.statuses["failed"][code] = 1

            if code != 404:
                self.traffic += int(response.headers['Content-Length'])
                return {'code': code, 'text': await response.text()}
            else:
                return {'code': code, 'text': ''}

    def page(self, release: str) -> str:
        """
        Возвращает адрес страницы по указанному типу релиза;

        :param release: тип релиза;
        :return: адрес страницы.
        """

        return (f'{self.url}/games/lib/release:asc/release_year:'
                f'released;'
                f'category:{self.releases[release]}')

    def setting(self, span: tuple, factor: int, threshold: int) -> None:
        """
        Настраивает менеджер;

        :param span: диапазон задержки;
        :param factor: масштаб задержки;
        :param threshold: порог смены типа задержки;
        :return: None.
        """

        self.delay.span = span
        self.delay.factor = factor
        self.delay.threshold = threshold

    def json(self) -> dict:
        """
        Возвращает текущие параметры:

        - span: Диапазон задержки;
        - factor: Масштаб задержки;
        - threshold: Порог смены типа задержки.

        :return: Текущие параметры.
        """

        return {'span': self.delay.span,
                'factor': self.delay.factor,
                'threshold': self.delay.threshold}
