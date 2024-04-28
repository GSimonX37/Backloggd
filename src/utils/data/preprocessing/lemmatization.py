from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


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
            tag = pos_tag([word])[0][1][0].lower()
            lemma = lemmatizer.lemmatize(word, tag)
            result.append(lemma)
            words[text] = lemma

    result = ' '.join(result)

    return result
