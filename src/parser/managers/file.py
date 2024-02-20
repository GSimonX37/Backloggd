import csv
import json
import os

from config.parser.managers.file import FIELD_NAMES
from config.paths import CHECKPOINT_PATH
from config.paths import FILE_RAW_PATH


class FileManager(object):
    """
    Файловый менеджер, задачами которого являются:

    - запись собранных файлов в csv файл;
    - учет количества собранных данных;
    - чтение и запись контрольной точки в формате json;

    :var file: имя файла с данными;
    :var checkpoint: имя файла контрольной точки в формате json;
    :var size: размер файла с данными;
    :var records: количество собранных данных.
    """

    def __init__(self):
        self.file: str | None = ''
        self.checkpoint: str | None = ''
        self.size: int | None = None
        self.records: int | None = None

    def create(self) -> None:
        """
        Создает csv-файл с данными;

        :return: None.
        """

        path = fr'{FILE_RAW_PATH}\{self.file}'
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(FIELD_NAMES)

        self.size, self.records = 0, 0

    async def write(self, records: list[list]) -> None:
        """
        Записывает данные в csv-файл. Учитывает количество собранных данных;

        :param records: записываемые данные;
        :return: None.
        """

        path = fr'{FILE_RAW_PATH}\{self.file}'
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')

            for record in records:
                writer.writerow(record)

        self.size = os.path.getsize(path)
        self.records += len(records)

    async def save(self, checkpoint: dict) -> None:
        """
        Записывает контрольную точки в формат json;

        :param checkpoint: контрольная точка;
        :return: None.
        """

        path = fr'{CHECKPOINT_PATH}\{self.checkpoint}'
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

        path = fr'{CHECKPOINT_PATH}\{checkpoint}'
        with open(path, 'r') as file:
            return json.loads(file.read())

    def delete(self) -> None:
        """
        Удаляет контрольную точки в формате json;

        :return: None.
        """

        path = fr'{CHECKPOINT_PATH}\{self.checkpoint}'
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def setting(self, file: str, mode: str, checkpoint: str | None) -> None:
        """
        Настраивает менеджер;

        :param file: имя файла с данными;
        :param mode: режим работы с файлом;
        :param checkpoint: имя файла контрольной точки в формате json;
        :return: None.
        """

        self.file = file
        self.checkpoint = checkpoint

        if mode == 'w':
            self.create()
        elif mode == 'a':
            path = fr'{FILE_RAW_PATH}\{file}'
            with open(path, 'r', newline='', encoding='utf-8') as file:
                rows = csv.reader(file, delimiter=',')
                self.records = sum([1 for _ in rows]) - 1

            self.size = os.path.getsize(path)

    def json(self) -> dict:
        """
        Возвращает текущие параметры:

        - имя файла с данными;

        :return: текущие параметры.
        """

        return {'file': self.file}
