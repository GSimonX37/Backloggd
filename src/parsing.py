import asyncio
import os

from config.parser.spider import SETTINGS
from config.paths import CHECKPOINT_PATH
from config.paths import FILE_RAW_PATH
from parser.parser import Parser
from utils.explorer import explorer


async def main():
    """
    Тока входа сбора данных;

    :return: None.
    """

    parser = Parser()

    os.system('cls')
    print('Соединение с сервером...', end=' ', flush=True)

    if code := await parser.connect() == 200:
        print('Ок.', flush=True)

        if names := explorer(CHECKPOINT_PATH, '*.json'):
            print(flush=True)
            print('Список контрольных точек:', names, sep='\n', flush=True)
            if checkpoint := input('Загрузить контрольную точку: '):
                await parser.load(checkpoint)
            else:
                print(flush=True)
                names = explorer(FILE_RAW_PATH, '*.csv')
                print('Список файлов:', names, sep='\n', flush=True)
                data = input('Укажите имя файла: ')

                settings = {}
                settings |= SETTINGS
                settings |= {'file': data}
                settings |= {'checkpoint': data.split('.')[0] + '.json'}
                await parser.setting(**settings)
        else:
            print(flush=True)
            names = explorer(FILE_RAW_PATH, '*.csv')
            print('Список файлов:', names, sep='\n', flush=True)
            data = input('Укажите имя файла: ')

            settings = {}
            settings |= SETTINGS
            settings |= {'file': data}
            settings |= {'checkpoint': data.split('.')[0] + '.json'}
            await parser.setting(**settings)

        await parser.state()

        print(flush=True)
        if not input('Нажмите "Enter" для продолжения.'):
            await parser.scrape()
    else:
        print(f'Неудача (код {code}).', end='\n\n', flush=True)

    await parser.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
