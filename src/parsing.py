import asyncio
import os

from config.parser.parser import SETTINGS
from config.paths import PATH_CHECKPOINT
from config.paths import PATH_RAW_DATA
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

        if names := explorer(PATH_CHECKPOINT, '*.json'):
            print(flush=True)
            print('Список контрольных точек:', names, sep='\n', flush=True)
            if checkpoint := input('Загрузить контрольную точку: '):
                await parser.load(checkpoint)
            else:
                print(flush=True)
                names = explorer(PATH_RAW_DATA, exclude=('checkpoints', ))
                print('Список директорий:', names, sep='\n', flush=True)
                data = input('Укажите имя директории: ')

                settings = {}
                settings |= SETTINGS
                settings |= {'directory': data}
                settings |= {'checkpoint': data.split('.')[0] + '.json'}
                await parser.setting(**settings)
        else:
            print(flush=True)
            names = explorer(PATH_RAW_DATA, exclude=('checkpoints',))
            print('Список директорий:', names, sep='\n', flush=True)
            data = input('Укажите имя директории: ')

            settings = {}
            settings |= SETTINGS
            settings |= {'directory': data}
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
