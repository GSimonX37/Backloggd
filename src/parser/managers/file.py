import csv
import json
import os
import pathlib

from config.parser.managers.file import FIELD_NAMES
from config.paths import PATH_CHECKPOINT
from config.paths import PATH_RAW_DATA


class FileManager(object):
    """
    Файловый менеджер, задачами которого являются:

    - запись собранных файлов в csv файл;
    - учет количества собранных данных;
    - чтение и запись контрольной точки в формате json;

    :var directory: имя директории с данными;
    :var checkpoint: имя файла контрольной точки в формате json;
    :var size: размер файла с данными;
    :var records: количество собранных данных.
    """

    def __init__(self):
        self.directory: str | None = ''
        self.checkpoint: str | None = ''
        self.size: dict = {file: 0 for file in FIELD_NAMES}
        self.records: dict = {file: 0 for file in FIELD_NAMES}
        self.image: dict = {
            'size': 0,
            'count': 0
        }

    def create(self) -> None:
        """
        Создает csv-файл с данными;

        :return: None.
        """

        path = fr'{PATH_RAW_DATA}\{self.directory}'

        if not os.path.exists(path):
            os.mkdir(path)

        for name, fields in FIELD_NAMES.items():
            name = fr'{path}\{name}.csv'
            with open(name, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(fields)

        path = fr'{PATH_RAW_DATA}\{self.directory}\posters'

        if not os.path.exists(path):
            os.mkdir(path)
        else:
            directory = pathlib.Path(path)
            for file in directory.iterdir():
                os.remove(file)

    async def write(self, records: dict) -> None:
        """
        Записывает данные в csv-файлы. Учитывает количество собранных данных;

        :param records: записываемые данные;
        :return: None.
        """

        for name in FIELD_NAMES:
            path = fr'{PATH_RAW_DATA}\{self.directory}\{name}.csv'
            with open(path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=',')
                for record in records[name]:
                    writer.writerow(record)
            self.size[name] = os.path.getsize(path)
            self.records[name] += len(records[name])

    async def images(self, images: dict):
        """
        Записывает изображения. Учитывает количество и размер изображений;

        :param images: записываемые изображения;
        :return: None.
        """

        for name, data in images.items():
            path = fr'{PATH_RAW_DATA}\{self.directory}\posters\{name}.jpg'
            with open(path, 'wb') as file:
                file.write(data)

            self.image['size'] += os.path.getsize(path)
        self.image['count'] += len(images)

    async def save(self, checkpoint: dict) -> None:
        """
        Записывает контрольную точки в формат json;

        :param checkpoint: контрольная точка;
        :return: None.
        """

        path = fr'{PATH_CHECKPOINT}\{self.checkpoint}'
        with open(path, 'w') as json_file:
            checkpoint = self.json() | checkpoint
            json_file.write(json.dumps(checkpoint, indent=4))

    @staticmethod
    def load(checkpoint: str) -> dict:
        """
        Читает контрольную точки в формате json;

        :param checkpoint: контрольная точка в формате json;
        :return: Статус.
        """

        path = fr'{PATH_CHECKPOINT}\{checkpoint}'
        with open(path) as file:
            return json.loads(file.read())

    def delete(self) -> None:
        """
        Удаляет контрольную точки в формате json;

        :return: None.
        """

        path = fr'{PATH_CHECKPOINT}\{self.checkpoint}'
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def setting(self,
                directory: str,
                mode: str,
                checkpoint: str | None) -> None:
        """
        Настраивает менеджер;

        :param directory: имя файла с данными;
        :param mode: режим работы с файлом;
        :param checkpoint: имя файла контрольной точки в формате json;
        :return: None.
        """

        self.directory = directory
        self.checkpoint = checkpoint

        if mode == 'w':
            self.create()
        elif mode == 'a':
            for name in FIELD_NAMES:
                path = fr'{PATH_RAW_DATA}\{self.directory}\{name}.csv'
                with open(path, newline='', encoding='utf-8') as file:
                    rows = csv.reader(file, delimiter=',')
                    self.records[name] = sum([1 for _ in rows]) - 1
                self.size[name] = os.path.getsize(path)

            path = fr'{PATH_RAW_DATA}\{self.directory}\posters'
            directory = pathlib.Path(path)
            for file in directory.iterdir():
                self.image['size'] += os.path.getsize(file)
                self.image['count'] += 1

    def json(self) -> dict:
        """
        Возвращает текущие параметры:

        - имя файла с данными;

        :return: текущие параметры.
        """

        return {'directory': self.directory}
