import re

import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')


def is_ascii(row: pd.Series) -> bool:
    """
    Проверяет, является ли значение в поле "description" пустым
    или содержит строку только с ascii символами;

    :param row: строка набора данных;
    :return: логическое значение.
    """

    return (pd.isna(row["description"]) or pd.notna(row["description"])
            and row["description"].isascii())


def clear(text: str) -> str:
    """
    Очищает строку от символов, отличных от букв английского алфавита
    в верхнем и нижнем регистрах;

    :param text: строка;
    :return: очищенная строка.
    """

    text = re.sub(
        pattern=r'[^a-zA-Z]',
        repl=' ',
        string=text
    )

    text = ' '.join(text.split()).lower()

    return text


def cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает строки набора данных от символов,
    отличных от букв английского алфавита в верхнем и нижнем регистрах;

    :param data: набор данных со строками;
    :return: очищенный набор данных.
    """

    data.iloc[:, 0] = data.iloc[:, 0].apply(clear)

    return data


def get_wordnet_pos(word: str) -> str:
    """
    Определяет часть речи слова;

    :param word: слово;
    :return: токен части речи.
    """

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


lemmatizer = WordNetLemmatizer()
words = {}


def lemmatize(text: str) -> str:
    """
    Производить лемматизацию строки;

    :param text: строка;
    :return: строка.
    """

    result = []

    for word in text.split():
        if lemma := words.get(word):
            result.append(lemma)
        else:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            result.append(lemma)
            words[text] = lemma

    result = ' '.join(result)

    return result


def lemmatization(data: np.ndarray) -> pd.DataFrame:
    """
    Производит лемматизацию строк набора данных;

    :param data: набор данных со строками;
    :return: набор данных.
    """

    data = pd.DataFrame(data)
    data.iloc[:, 0] = data.iloc[:, 0].apply(lemmatize)

    return data
