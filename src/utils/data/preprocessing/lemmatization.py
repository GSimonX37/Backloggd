import nltk

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


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
