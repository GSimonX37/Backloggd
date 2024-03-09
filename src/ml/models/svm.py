import numpy as np
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

nltk.download('stopwords')

title = 'LinearSVC'

vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words=stopwords.words('english')
)

standardizer = Pipeline(
    steps=[
        ('vectorizer', vectorizer)
    ]
)

estimator = LinearSVC(
    penalty='l2',
    loss='squared_hinge',
    random_state=42
)

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
    'standardizer__vectorizer__norm': [None, 'l2'],
    'standardizer__vectorizer__sublinear_tf': [True],
    'standardizer__vectorizer__max_features': np.arange(
        start=750_000,
        stop=1_000_001,
        step=250_000
    ).tolist(),
    'estimator__estimator__tol': [0.0001, 0.001, 0.01],
    'estimator__estimator__dual': [True, False],
    'estimator__estimator__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}
