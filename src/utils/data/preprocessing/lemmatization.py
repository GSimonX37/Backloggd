from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


lemmatizer = WordNetLemmatizer()
words = {}


def get_tag(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


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
            tag = get_tag(word)
            lemma = lemmatizer.lemmatize(word, tag)
            result.append(lemma)
            words[text] = lemma

    result = ' '.join(result)

    return result
