import pathlib


def explorer(path: str, ext: str = '') -> str:
    """
    Формирует и нумерует список директорий и файлов по указанному пути
    в указанном формате;

    :param path: путь к файлам или папкам;
    :param ext: формат файлов;
    :return: список директорий и файлов в указанном формате.
    """

    directory = pathlib.Path(path)

    elements = []

    if ext:
        for i, element in enumerate(directory.glob(ext), start=1):
            if element.is_file():
                elements += [f'{i}. {element.name}.']
    else:
        for i, element in enumerate(directory.iterdir(), start=1):
            if element.is_dir():
                elements += [f'{i}. {element.name}.']

    return '\n'.join(elements)
