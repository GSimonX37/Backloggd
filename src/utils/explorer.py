import pathlib


def explorer(path: str, ext: str = '', exclude: tuple = ()) -> str:
    """
    Формирует и нумерует список директорий и файлов по указанному пути
    в указанном формате;

    :param path: путь к файлам или папкам;
    :param ext: формат файлов;
    :param exclude: исключения;
    :return: список директорий и файлов в указанном формате.
    """

    directory = pathlib.Path(path)

    elements = []

    if ext:
        for i, element in enumerate(directory.glob(ext), start=1):
            if element.is_file() and element.name not in exclude:
                elements += [f'{i}. {element.name}.']
    else:
        i = 1
        for element in directory.iterdir():
            if element.is_dir() and element.name not in exclude:
                elements += [f'{i}. {element.name}.']
                i += 1

    return '\n'.join(elements)
