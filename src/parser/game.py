class Game(object):
    """
    Видеоигра;

    :var id: идентификатор;
    :var image: ссылка на постер;
    :var name: название;
    :var date: дата выхода;
    :var genres: игровые жанры;
    :var platforms: игровые платформы;
    :var developers: группа разработчиков;
    :var rating: средний рейтинг;
    :var scores: количество оценок пользователей по рейтингам;
    :var reviews: количество отзывов пользователей;
    :var plays: общее количество игроков;
    :var playing: количество игроков в настоящий момент;
    :var backlogs: количество добавлений в backlog;
    :var wishlists: количество добавлений в wishlist;
    :var description: описание.
    """

    def __init__(self, num: int):
        self.id: int | None = num
        self.image: str | None = None
        self.name: str | None = None
        self.date: str | None = None
        self.genres: list | None = None
        self.platforms: list[str] | None = None
        self.developers: list[str] | None = None
        self.rating: float | None = None
        self.scores: list[int] | None = None
        self.reviews: int | None = None
        self.plays: int | None = None
        self.playing: int | None = None
        self.backlogs: int | None = None
        self.wishlists: int | None = None
        self.description: str | None = None

    def __bool__(self):
        for attribute in self.__dict__:
            if attribute != 'release' and getattr(self, attribute):
                return True

        return False

    def csv(self) -> dict:
        """
        Возвращает данные для записи в csv-файл.

        :return: Данные для записи в csv-файл.
        """
        scores = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        csv = {
            'games': [
                self.id,
                self.name,
                self.date,
                self.rating,
                self.reviews,
                self.plays,
                self.playing,
                self.backlogs,
                self.wishlists,
                self.description
            ],
            'developers': [[self.id, genre] for genre in self.developers],
            'platforms': [[self.id, genre] for genre in self.platforms],
            'genres': [[self.id, genre] for genre in self.genres],
            'scores': [[self.id, score, amount]
                       for score, amount
                       in zip(scores, self.scores)]
        }

        return csv
