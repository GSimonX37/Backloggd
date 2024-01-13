import string

import pandas as pd


letters = set(string.ascii_letters)
digits = set(string.digits)
punctuation = set(string.punctuation)


def digits_number(text: str) -> int:
    """
    Вычисляет количество цифр в строке;

    :param text: строка;
    :return: количество цифр в строке.
    """

    return len([t for t in text if t in digits])


def letters_number(text: str) -> int:
    """
    Вычисляет количество букв в строке;

    :param text: строка;
    :return: количество букв в строке.
    """

    return len([t for t in text if t in letters])


def lower_words_number(text: str) -> int:
    """
    Вычисляет количество слов в нижнем регистре в строке;

    :param text: строка;
    :return: количество слов в нижнем регистре в строке.
    """

    return len([t for t in text if t.islower()])


def punctuation_mark_number(text: str) -> int:
    """
    Вычисляет количество знаков пунктуации в строке;

    :param text: строка;
    :return: количество знаков пунктуации в строке.
    """

    return len([t for t in text if t in punctuation])


def title_words_number(text: str) -> int:
    """
    Вычисляет количество слов с заглавной буквы в строке;

    :param text: строка;
    :return: количество слов с заглавной буквы в строке.
    """

    return len([t for t in text if t.istitle()])


def upper_words_number(text: str) -> int:
    """
    Вычисляет количество слов в верхнем регистре в строке;

    :param text: строка;
    :return: количество слов в верхнем регистре в строке.
    """

    return len([t for t in text if t.isupper()])


def words_number(text: str) -> int:
    """
    Вычисляет количество слов в строке;

    :param text: строка;
    :return: количество слов в строке.
    """

    return len(text.split())


def sentences_number(text: str) -> int:
    """
    Вычисляет количество предложений в строке;

    :param text: строка;
    :return: количество предложений в строке.
    """

    return text.count('.')


def generate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует для текста (параметр data) следующие признаки:

    - symbols_number: количество символов;
    - letters_number: количество букв;
    - digits_number: количество цифр;
    - punctuation_mark_number: количество знаков пунктуации;
    - words_number: количество слов;
    - title_words_number: количество слов с заглавной буквы;
    - upper_words_number: количество слов в верхнем регистре;
    - lower_words_number: количество слов в нижнем регистре;
    - sentences_number: количество предложений;
    - letter_symbol_ratio: отношение количества букв к количеству символов;
    - digits_symbol_ratio:отношение количества цифр к количеству символов;
    - punctuation_mark_symbol_ratio: отношение количества знаков пунктуации
      к количеству символов;
    - sentences_words_ratio: отношение количества предложений
      к количеству слов;
    - title_words_ratio: отношение количества слов,
      начинающихся с заглавной буквы к количеству слов;
    - upper_words_ratio: отношение количества слов в верхнем регистре
      к количеству слов;
    - lower_words_ratio: отношение количества слов в нижнем регистре
      к количеству слов;

    :param data: строка, для которой будут сгенерированы признаки;
    :return: новые признаки.
    """

    features = pd.DataFrame(
        data=data.values,
        columns=['description']
    )

    features.insert(
        loc=features.shape[1],
        column='symbols_number',
        value=features.iloc[:, 0].apply(len)
    )
    features.insert(
        loc=features.shape[1],
        column='letters_number',
        value=features.iloc[:, 0].apply(letters_number)
    )
    features.insert(
        loc=features.shape[1],
        column='digits_number',
        value=features.iloc[:, 0].apply(digits_number)
    )
    features.insert(
        loc=features.shape[1],
        column='punctuation_mark_number',
        value=features.iloc[:, 0].apply(punctuation_mark_number)
    )
    features.insert(
        loc=features.shape[1],
        column='words_number',
        value=features.iloc[:, 0].apply(words_number)
    )
    features.insert(
        loc=features.shape[1],
        column='title_words_number',
        value=features.iloc[:, 0].apply(title_words_number)
    )
    features.insert(
        loc=features.shape[1],
        column='upper_words_number',
        value=features.iloc[:, 0].apply(upper_words_number)
    )
    features.insert(
        loc=features.shape[1],
        column='lower_words_number',
        value=features.iloc[:, 0].apply(lower_words_number)
    )
    features.insert(
        loc=features.shape[1],
        column='sentences_number',
        value=features.iloc[:, 0].apply(sentences_number)
    )

    features.insert(
        loc=features.shape[1],
        column='letter_symbol_ratio',
        value=features['letters_number'] / features['symbols_number']
    )
    features.insert(
        loc=features.shape[1],
        column='digits_symbol_ratio',
        value=features['digits_number'] / features['symbols_number']
    )
    features.insert(
        loc=features.shape[1],
        column='punctuation_mark_symbol_ratio',
        value=features['punctuation_mark_number'] / features['symbols_number']
    )
    features.insert(
        loc=features.shape[1],
        column='sentences_words_ratio',
        value=features['sentences_number'] / features['words_number']
    )

    features.insert(
        loc=features.shape[1],
        column='title_words_ratio',
        value=features['title_words_number'] / features['words_number']
    )
    features.insert(
        loc=features.shape[1],
        column='upper_words_ratio',
        value=features['upper_words_number'] / features['words_number']
    )
    features.insert(
        loc=features.shape[1],
        column='lower_words_ratio',
        value=features['lower_words_number'] / features['words_number']
    )

    return features
