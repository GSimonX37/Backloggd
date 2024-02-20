import datetime


class Game(object):
    """
    Видеоигра;

    :var name: название;
    :var date: дата выхода;
    :var genres: игровые жанры;
    :var platforms: игровые платформы;
    :var developers: группа разработчиков;
    :var rating: средний рейтинг;
    :var votes: количество оценок пользователей;
    :var release: тип релиза;
    :var reviews: количество отзывов пользователей;
    :var plays: общее количество игроков;
    :var playing: количество игроков в настоящий момент;
    :var backlogs: количество добавлений в backlog;
    :var wishlists: количество добавлений в wishlist;
    :var description: описание.
    """

    def __init__(self, release: str):
        self.name: str | None = None
        self.date: datetime.date | None = None
        self.genres: list | None = None
        self.platforms: list[str] | list = []
        self.developers: list[str] | list = []
        self.rating: float | None = None
        self.votes: list[int] | list = []
        self.release: str = release
        self.reviews: int | None = None
        self.plays: int | None = None
        self.playing: int | None = None
        self.backlogs: int | None = None
        self.wishlists: int | None = None
        self.description: str | None = None

    def __bool__(self):
        for attribute in self.__dict__:
            if attribute != 'release':
                if getattr(self, attribute):
                    return True

        return False

    def csv(self) -> list:
        """
        Возвращает данные для записи в csv-файл.

        :return: Данные для записи в csv-файл.
        """

        return [
            self.name,
            self.date,
            self.developers,
            self.rating,
            self.votes,
            self.platforms,
            self.genres,
            self.release,
            self.reviews,
            self.plays,
            self.playing,
            self.backlogs,
            self.wishlists,
            self.description
        ]
