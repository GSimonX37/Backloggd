from functools import cache

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


@cache
def get_tag(word) -> str:
    """
    Присваивает токен части речи слову;

    :param word: слово;
    :return: токен части речи слова.
    """

    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"A": wordnet.ADJ,
                "S": wordnet.ADJ_SAT,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


@cache
def lemma(word: str, tag) -> str:
    """
    Производит лемматизацию слова;

    :param word: слово;
    :param tag: токен части речи слова;
    :return: слово.
    """

    return lemmatizer.lemmatize(word, tag)


def lemmatize(text: str) -> str:
    """
    Производит лемматизацию строки;

    :param text: строка;
    :return: строка.
    """

    result = []

    for word in text.split():
        tag = get_tag(word)
        word = lemma(word, tag)
        result.append(word)

    result = ' '.join(result)

    return result
