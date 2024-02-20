import asyncio

from config.parser.spider import RELEASES
from config.parser.spider import VALID_ATTEMPTS
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
    :var releases: релизы видеоигр;
    :var stopped: флаг остановки трансфера данных.
    """

    def __init__(self):
        self.file: FileManager = FileManager()
        self.network: NetworkManager = NetworkManager()
        self.output: OutputManager = OutputManager()
        self.parsing: ParsingManager = ParsingManager()
        self.progress: ProgressManager = ProgressManager()
        self.releases: dict[str, str] = RELEASES
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

        for release, (current, last) in self.progress.progress.items():
            for page in range(current, last + 1):
                link = self.network.page(release)
                links = await self.links(link, page)
                games = await self.games(links, release)

                data = [game.csv() for game in games if game]
                await self.file.write(data)

                await self.progress.next()

                await self.save()

        self.file.delete()

        await self.transfer()
        await self.output.state()

        self.output.stopped = True
        self.stopped = True

    async def links(self, link: str, page: int) -> list[str]:
        """
        Получает ссылки на страницы с данными;

        :param link: страница с играми;
        :param page: номер страницы;
        :return: ссылки на страницы с данными.
        """

        code, links = None, []

        while code != 200:
            response = await self.network.get(link, params={'page': f'{page}'})
            code = response['code']

            if code == 200:
                text = response['text']
                links = await self.parsing.games(self.network.url, text)

        return links

    async def page(self, link: str) -> None | int:
        """
        Получает номер последней страницы по указанной ссылке;

        :param link: ссылка на страницу с данными;
        :return: номер последней страницы.
        """

        code, number, attempts = None, None, 0

        while code != 200 and attempts < VALID_ATTEMPTS:
            response = await self.network.get(link)
            code, attempts = response['code'], attempts + 1

            if code == 200:
                text = response['text']
                number = await self.parsing.page(text)

        return number

    async def pages(self, releases: tuple[str]) -> tuple[int]:
        """
        Получает номера последних страниц по указанным ссылкам;

        :param releases: релизы видеоигр;
        :return: последние страницы.
        """

        tasks = []

        for release in releases:
            release = self.network.releases[release]
            link = self.network.url
            link += (f'/games/lib/release:asc/'
                     f'release_year:released;'
                     f'category:{release}')
            tasks.append(asyncio.create_task(self.page(link)))

        numbers = await asyncio.gather(*tasks)

        return numbers

    async def game(self, link: str, release: str) -> Game:
        """
        Получает данные о видеоигре по указанной ссылке;

        :param link: ссылка на страницу с данными;
        :param release: тип релиза видеоигры;
        :return: данные видеоигры.
        """

        code, attempts = None, 0
        game = Game(release)

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
                        return game

                    response = await self.network.get(link)
                    code = response['code']

                    attempts += 1

                    if code == 200:
                        text = response['text']
                        await self.parsing.statistic(game, text)

        return game

    async def games(self, links: list[str], release: str) -> tuple[Game]:
        """
        Получает данные о видеоиграх по указанным ссылкам;

        :param links: ссылки на страницы с данными;
        :param release: тип релиза видеоигр;
        :return: данные видеоигр.
        """

        tasks = []
        for link in links:
            tasks.append(asyncio.create_task(self.game(link, release)))

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
                      file: str,
                      mode: str,
                      timeout: int,
                      checkpoint: str):
        """
        Настраивает менеджеры;

        :param span: диапазон задержки;
        :param factor: масштаб задержки;
        :param threshold: порог смены типа задержки;
        :param file: имя файла с данными;
        :param mode: режим работы с файлом;
        :param timeout: задержка между выводами текущего состояния;
        :param checkpoint: имя файла контрольной точки в формате json;
        :return: None.
        """

        self.network.setting(span, factor, threshold)
        self.file.setting(file, mode, checkpoint)

        progress = {}
        latest = await self.pages((*self.releases.keys(), ))
        for release, last in zip([*self.releases.keys()], latest):
            progress[release] = [1, last]

        self.progress.setting(progress)
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
            file=settings['file'],
            mode='a',
            checkpoint=checkpoint
        )

        self.network.setting(
            span=settings['span'],
            factor=settings['factor'],
            threshold=settings['threshold']
        )

        progress = {}
        latest = await self.pages((*self.releases.keys(), ))
        pages = zip(settings['progress'].items(), latest)
        for (release, current), last in pages:
            progress[release] = [current, last]

        self.progress.setting(
            progress=progress
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
                file=self.file.file,
                size=self.file.size,
                records=self.file.records,
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
