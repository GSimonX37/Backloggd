import nltk
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


nltk.download('stopwords')

title = 'ComplementNB'

vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words=stopwords.words('english')
)

standardizer = Pipeline(
    steps=[
        ('vectorizer', vectorizer)
    ]
)

estimator = ComplementNB()
estimator = MultiOutputClassifier(
    estimator=estimator,
    n_jobs=4
)

model = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator)
    ]
)

params = {
    'standardizer__vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'standardizer__vectorizer__norm': [None, 'l1', 'l2'],
    'standardizer__vectorizer__sublinear_tf': [True, False],
    'standardizer__vectorizer__max_features': np.arange(
        start=500_000,
        stop=1_000_001,
        step=250_000
    ).tolist(),
    'estimator__estimator__norm': [False, True],
    'estimator__estimator__alpha': np.linspace(
        start=1.0,
        stop=10.0,
        num=10
    ).round(5).tolist()
}
