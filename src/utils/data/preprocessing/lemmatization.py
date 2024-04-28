from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

from functools import cache

lemmatizer = WordNetLemmatizer()


@cache
def get_tag(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


@cache
def lemma(word: str, tag) -> str:
    return lemmatizer.lemmatize(word, tag)


def lemmatize(text: str) -> str:
    """
    Производить лемматизацию строки;

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
