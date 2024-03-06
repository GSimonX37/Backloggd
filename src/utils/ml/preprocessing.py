import re

import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = stopwords.words('english')


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


def cleaning(data: pd.Series) -> pd.Series:
    """
    Очищает строки набора данных от символов,
    отличных от букв английского алфавита в верхнем и нижнем регистрах;

    :param data: набор данных со строками;
    :return: очищенный набор данных.
    """

    return data.copy().apply(clear)


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


def lemmatization(data: pd.Series) -> pd.Series:
    """
    Производит лемматизацию строк набора данных;

    :param data: набор данных со строками;
    :return: набор данных.
    """

    return data.copy().apply(lemmatize)
