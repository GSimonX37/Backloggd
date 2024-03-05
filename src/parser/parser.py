import asyncio

from config.parser.parser import VALID_ATTEMPTS
from parser.game import Game
from parser.managers.file import FileManager
from parser.managers.network.network import NetworkManager
from parser.managers.output import OutputManager
from parser.managers.parsing import ParsingManager
from parser.managers.progress import ProgressManager


class Parser(object):
    """
    Программа, осуществляющая сбор, обработку и хранение данных;

    :var file: файловый менеджер;
    :var network: сетевой менеджер;
    :var output: менеджер вывода;
    :var parsing: менеджер парсинга;
    :var progress: менеджер прогресса;
    :var stopped: флаг остановки трансфера данных.
    """

    def __init__(self):
        self.file: FileManager = FileManager()
        self.network: NetworkManager = NetworkManager()
        self.output: OutputManager = OutputManager()
        self.parsing: ParsingManager = ParsingManager()
        self.progress: ProgressManager = ProgressManager()
        self.stopped: bool = False

    async def connect(self) -> int:
        """
        Проверяет соединение с сервером перед началом сбора данных;

        :return: Код статуса запроса.
        """

        code = await self.network.connect()

        return code

    async def scrape(self) -> None:
        """
        Запускает:

        - процесс сбора данных;
        - трансфер менеджеру вывода параметров остальных менеджеров;
        - отображение текущего состояния сбора данных;

        :return: None.
        """

        self.progress.starting()

        tasks = [
            asyncio.create_task(self.run()),
            asyncio.create_task(self.transfer(True)),
            asyncio.create_task(self.output.state(True))
        ]

        await asyncio.gather(*tasks)

    async def run(self) -> None:
        """
        Запускает процесс сбора данных;

        :return: None
        """

        current, last = self.progress.progress

        for page in range(current, last + 1):
            links = await self.links(page)

            first = (page - 1) * 36 + 1
            ids = [10 ** 6 + i for i in range(first, first + len(links))]

            games = await self.games(links, ids)

            data = {
                'games': [],
                'developers': [],
                'platforms': [],
                'genres': [],
                'scores': []
            }

            for game in games:
                csv = game.csv()

                data['games'].append(csv['games'])
                data['developers'] += csv['developers']
                data['platforms'] += csv['platforms']
                data['genres'] += csv['genres']
                data['scores'] += csv['scores']

            await self.file.write(data)

            images = {game.id: game.image for game in games if game.image}
            await self.file.images(images)

            await self.progress.next()

            await self.save()

        self.file.delete()

        await self.transfer()
        await self.output.state()

        self.output.stopped = True
        self.stopped = True

    async def links(self, page: int) -> list[str]:
        """
        Получает ссылки на страницы с данными;

        :param page: номер страницы;
        :return: ссылки на страницы с данными.
        """

        code, links = None, []

        while code != 200:
            link = f'{self.network.url}/games/lib/release:asc/'
            response = await self.network.get(link, params={'page': f'{page}'})
            code = response['code']

            if code == 200:
                text = response['text']
                links = await self.parsing.games(self.network.url, text)

        return links

    async def page(self) -> None | int:
        """
        Получает номер последней страницы по указанной ссылке;

        :return: номер последней страницы.
        """

        code, number, attempts = None, None, 0
        link = f'{self.network.url}/games/lib/release:asc/'
        while code != 200 and attempts < VALID_ATTEMPTS:
            response = await self.network.get(link)
            code, attempts = response['code'], attempts + 1

            if code == 200:
                text = response['text']
                number = await self.parsing.page(text)

        return number

    async def game(self, link: str, i: int) -> Game:
        """
        Получает данные о видеоигре по указанной ссылке;

        :param link: ссылка на страницу с данными;
        :param i: id видеоигры;
        :return: данные видеоигры.
        """

        code, attempts = None, 0
        game = Game(i)

        while code != 200:
            if attempts > VALID_ATTEMPTS and code != 429:
                return game

            response = await self.network.get(link)
            code = response['code']

            attempts += 1

            if code == 200:
                text = response['text']
                await self.parsing.parse(game, text)

                code, link = None, link.replace('games', 'logs') + 'plays/'
                while code != 200:
                    if attempts > VALID_ATTEMPTS and code != 429:
                        break

                    response = await self.network.get(link)
                    code = response['code']

                    attempts += 1

                    if code == 200:
                        text = response['text']
                        await self.parsing.statistic(game, text)

                if game.image:
                    code, attempts = None, 0

                    while code != 200:
                        if attempts > VALID_ATTEMPTS and code != 429:
                            game.image = None
                            return game

                        response = await self.network.get(game.image)
                        code = response['code']

                        attempts += 1

                        if code == 200:
                            game.image = response['binary']

        return game

    async def games(self, links: list[str], ids: list[int]) -> tuple[Game]:
        """
        Получает данные о видеоиграх по указанным ссылкам;

        :param links: ссылки на страницы с данными;
        :param ids: идентификаторы видеоигр;
        :return: данные видеоигр.
        """

        tasks = []
        for link, num in zip(links, ids):
            tasks.append(asyncio.create_task(self.game(link, num)))

        games = await asyncio.gather(*tasks)

        return games

    async def disconnect(self) -> None:
        """
        Закрывает сессию;

        :return: None.
        """

        await self.network.session.close()

    async def setting(self,
                      span: tuple[int, int],
                      factor: int,
                      threshold: int,
                      directory: str,
                      mode: str,
                      timeout: int,
                      checkpoint: str):
        """
        Настраивает менеджеры;

        :param span: диапазон задержки;
        :param factor: масштаб задержки;
        :param threshold: порог смены типа задержки;
        :param directory: имя директории с данными;
        :param mode: режим работы с файлом;
        :param timeout: задержка между выводами текущего состояния;
        :param checkpoint: имя файла контрольной точки в формате json;
        :return: None.
        """

        self.network.setting(span, factor, threshold)
        self.file.setting(directory, mode, checkpoint)

        last = await self.page()

        self.progress.setting([1, last])
        self.output.setting(timeout)

        await self.transfer()

    async def save(self) -> None:
        """
        Записывает контрольную точки в формат json;

        :return: None.
        """

        settings = {}
        settings |= self.progress.json()
        settings |= self.network.json()
        settings |= self.output.json()

        await self.file.save(settings)

    async def load(self, checkpoint: str) -> None:
        """
        Читает контрольную точки в формате json;

        :param checkpoint: контрольная точка;
        :return: None.
        """

        settings = self.file.load(checkpoint)

        self.file.setting(
            directory=settings['directory'],
            mode='a',
            checkpoint=checkpoint
        )

        self.network.setting(
            span=settings['span'],
            factor=settings['factor'],
            threshold=settings['threshold']
        )

        last = await self.page()

        self.progress.setting(
            progress=[settings['progress'], last]
        )

        self.output.setting(
            timeout=settings['timeout']
        )

        await self.transfer()

    async def state(self) -> None:
        """
        Выводит текущее состояние на экран;

        :return: None.
        """

        await self.output.state()

    async def transfer(self, repeat: bool = False) -> None:
        """
        Трансфер менеджеру вывода параметров остальных менеджеров;

        :return: None.
        """

        while not self.stopped:
            await self.output.file(
                directory=self.file.directory,
                size=self.file.size,
                records=self.file.records,
                image=self.file.image
            )

            await self.output.network(
                statuses=self.network.statuses,
                traffic=self.network.traffic,
                span=self.network.delay.span,
            )

            await self.output.parsing(
                success=self.parsing.success,
                failed=self.parsing.failed,
            )

            await self.output.progress(
                passed=self.progress.passed(),
                finish=self.progress.finished,
                speed=self.progress.speed,
                interval=self.progress.interval
            )

            if not repeat:
                break

            await asyncio.sleep(1)
