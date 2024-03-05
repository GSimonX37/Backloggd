import datetime

from bs4 import BeautifulSoup

from config.parser.managers.parsing import PARSING_FIELDS
from parser.game import Game


class ParsingManager(object):
    """
    Менеджер парсинга, задачами которого являются:

    - парсинг полученных данных;
    - учет успешно и неуспешно спарсенных данных;

    :var success: успешно спарсенные данные;
    :var failed: неуспешно спарсенные данные.
    """

    def __init__(self):
        self.success: dict[str: int] = {field: 0 for field in PARSING_FIELDS}
        self.failed: dict[str: int] = {field: 0 for field in PARSING_FIELDS}

    async def parse(self, game: Game,  text: str) -> None:
        """
        Осуществляет парсинг основных данных;

        :param game: экземпляр класса Game;
        :param text: данные для парсинга;
        :return: None.
        """

        soup = BeautifulSoup(text, 'html.parser')
        soup = soup.find('div', class_='row', id='game-profile')

        game.image = await self.image(soup)
        game.name = await self.name(soup)
        game.date = await self.date(soup)
        game.developers = await self.developers(soup)
        game.rating = await self.rating(soup)
        game.scores = await self.scores(soup)
        game.reviews = await self.reviews(soup)
        game.platforms = await self.platforms(soup)
        game.genres = await self.genres(soup)
        game.description = await self.description(soup)

    async def image(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг ссылки на изображение;

        :param soup: объект BeautifulSoup;
        :return: ссылка на изображение.
        """

        try:
            html = soup.find('img', class_='card-img height')
            image = html['src']

            self.success['image'] += 1
            return image
        except AttributeError:
            self.failed['image'] += 1
        except ValueError:
            self.failed['image'] += 1
        except KeyError:
            self.failed['image'] += 1
        except TypeError:
            self.failed['image'] += 1

    async def name(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг названия видеоигры;

        :param soup: объект BeautifulSoup;
        :return: название видеоигры.
        """

        try:
            name = soup.find('h1', class_='mb-0').text
            self.success['name'] += 1
            return name
        except AttributeError:
            self.failed['name'] += 1
        except ValueError:
            self.failed['name'] += 1

    async def date(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг даты выхода видеоигры;

        :param soup: объект BeautifulSoup;
        :return: дата выхода видеоигры.
        """

        try:
            date = (soup
                    .find('div', class_='col-auto mt-auto pr-0')
                    .find('a')
                    .text)
            date = (datetime
                    .datetime
                    .strptime(date, '%b %d, %Y')
                    .strftime('%Y-%m-%d'))

            self.success['date'] += 1
            return date
        except AttributeError:
            self.failed['date'] += 1
        except ValueError:
            self.failed['date'] += 1

    async def developers(self, soup: BeautifulSoup) -> list[str]:
        """
        Осуществляет парсинг разработчиков видеоигры;

        :param soup: объект BeautifulSoup;
        :return: разработчики видеоигры.
        """

        try:
            companies = (soup
                         .find('div', class_='col-auto pl-lg-1 sub-title')
                         .find_all('a'))
            developers = [*map(lambda x: x.text, companies)]

            self.success['developers'] += 1
            return developers
        except AttributeError:
            self.failed['developers'] += 1
            return []
        except ValueError:
            self.failed['developers'] += 1
            return []

    async def rating(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг рейтинга видеоигры;

        :param soup: объект BeautifulSoup;
        :return: рейтинг видеоигры.
        """

        try:
            rating = float(soup
                           .find(class_='side-section')
                           .find('h1', class_='text-center')
                           .text)

            self.success['rating'] += 1
            return rating
        except AttributeError:
            self.failed['rating'] += 1
        except ValueError:
            self.failed['rating'] += 1

    async def scores(self, soup: BeautifulSoup) -> list[int]:
        """
        Осуществляет парсинг количества голосов пользователей;

        :param soup: объект BeautifulSoup;
        :return: голоса пользователей.
        """

        try:
            votes = (soup
                     .find(class_='side-section')
                     .find('div', id='ratings-bars-height')
                     .find_all('div'))
            votes = [vote['data-tippy-content'].split()[0]
                     for vote in votes[::2]]

            self.success['scores'] += 1
            return votes
        except AttributeError:
            self.failed['scores'] += 1
            return []
        except ValueError:
            self.failed['scores'] += 1
            return []

    async def reviews(self, soup: BeautifulSoup) -> int | None:
        """
        Осуществляет парсинг количества отзывов пользователей;

        :param soup: объект BeautifulSoup;
        :return: количество отзывов пользователей.
        """

        try:

            reviews = soup.find('div', id='center-content')
            reviews = (reviews
                       .findAll('p', class_='game-page-sidecard')[-1]
                       .text)
            reviews = reviews.split()[0]

            if 'K' in reviews:
                reviews = int(float(reviews.replace('K', '')) * 1000)
            else:
                reviews = int(reviews)

        except AttributeError:
            self.failed['reviews'] += 1
            return
        except ValueError:
            self.failed['reviews'] += 1
            return

        if reviews > 10:
            try:
                reviews = soup.find_all('a', class_='small-link')[-1].text
                reviews = int(reviews.split()[-2])
            except IndexError:
                self.failed['reviews'] += 1
                return
            except AttributeError:
                self.failed['reviews'] += 1
                return
            except ValueError:
                self.failed['reviews'] += 1
                return

        self.success['reviews'] += 1
        return reviews

    async def platforms(self, soup: BeautifulSoup) -> list[str]:
        """
        Осуществляет парсинг платформ видеоигры;

        :param soup: объект BeautifulSoup;
        :return: платформы видеоигры.
        """

        try:
            platforms = (soup
                         .find('div', class_='col-lg-4 mt-1 mt-lg-2 col-12')
                         .find_all('div')[2]
                         .find_all('a'))
            platforms = [*map(lambda x: x.text.strip(), platforms)]

            self.success['platforms'] += 1
            return platforms
        except AttributeError:
            self.failed['platforms'] += 1
            return []
        except ValueError:
            self.failed['platforms'] += 1
            return []

    async def genres(self, soup: BeautifulSoup) -> list[str]:
        """
        Осуществляет парсинг жанров видеоигры;

        :param soup: объект BeautifulSoup;
        :return: жанры видеоигры.
        """

        try:
            genres = (soup
                      .find('div', class_='col-lg-4 mt-1 mt-lg-2 col-12')
                      .find_all('div')[6]
                      .find_all('a'))
            genres = [*map(lambda x: x.text, genres)]

            self.success['genres'] += 1
            return genres
        except AttributeError:
            self.failed['genres'] += 1
            return []
        except ValueError:
            self.failed['genres'] += 1
            return []

    async def description(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг описания видеоигры;

        :param soup: объект BeautifulSoup;
        :return: описание видеоигры.
        """

        try:
            description = (soup
                           .find('div', id='center-content')
                           .find('div', id='collapseSummary')
                           .findAll('p'))
            description = '\n'.join([x.text for x in description])
            description = description.replace('\n', ' ')

            self.success['description'] += 1
            return description
        except AttributeError:
            self.failed['description'] += 1
            return
        except ValueError:
            self.failed['description'] += 1
            return

    async def statistic(self, game, text: str) -> None:
        """
        Осуществляет парсинг статистики;

        :param game: экземпляр класса Game;
        :param text: данные для парсинга;
        :return: None.
        """

        soup = BeautifulSoup(text, 'html.parser')

        try:
            statistic = soup.find('div', id='plays-nav').find_all('a')
            statistic = [*map(lambda x: x.text, statistic)]
            statistic = [int(s.split()[1]) for s in statistic]

            game.plays, game.playing, game.backlogs, game.wishlists = statistic

            self.success['plays'] += 1
            self.success['playing'] += 1
            self.success['backlogs'] += 1
            self.success['wishlists'] += 1
        except AttributeError:
            self.failed['plays'] += 1
            self.failed['playing'] += 1
            self.failed['backlogs'] += 1
            self.failed['wishlists'] += 1
        except ValueError:
            self.failed['plays'] += 1
            self.failed['playing'] += 1
            self.failed['backlogs'] += 1
            self.failed['wishlists'] += 1

    @staticmethod
    async def page(text: str) -> int:
        """
        Осуществляет парсинг номера последней страницы;

        :param text: данные для парсинга;
        :return: номер последней страницы;
        """

        soup = BeautifulSoup(text, 'html.parser')
        number = int(soup.find_all('span', class_='page')[-2].next.text)

        return number

    @staticmethod
    async def games(url: str, text: str) -> list:
        """
        Осуществляет парсинг номера последней страницы;

        :param url: адрес сайта backloggd;
        :param text: данные для парсинга;
        :return: номер последней страницы.
        """

        links = []

        soup = BeautifulSoup(text, 'html.parser')
        html = soup.find_all('a', class_='cover-link')
        for a in html:
            name = a['href'].split('/')[2]
            links.append(f'{url}/games/{name}/')

        return links
