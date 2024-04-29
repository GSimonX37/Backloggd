import re


def letters(text: str) -> str:
    """
    Очищает строку от символов, отличных от букв английского алфавита;

    :param text: строка;
    :return: очищенная строка.
    """

    return re.sub(
        pattern=r'[^a-zA-Z]',
        repl=' ',
        string=text
    )


def spaces(text: str) -> str:
    """
    Очищает строку последовательных пробельных символов;

    :param text: строка;
    :return: очищенная строка.
    """

    return re.sub(
        pattern=r'\s{1,}',
        repl=' ',
        string=text
    )


def length(text: str, size: int) -> str:
    """
    Заменяет последовательность размера менее "size" пробелом;

    :param text: строка;
    :param size: размер последовательности;
    :return: очищенная строка.
    """

    pattern = fr'(?<=[\s\b])(\w{{1,{size}}})(?=[\s\b])'

    return re.sub(
        pattern=pattern,
        repl=' ',
        string=text
    )
