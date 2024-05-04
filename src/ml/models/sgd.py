from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from .model import Model


def metric(y_true, y_predict) -> float:
    return f1_score(
        y_true=y_true,
        y_pred=y_predict,
        average='weighted'
    )


vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words=stopwords.words('english'),
    ngram_range=(1, 3)
)

standardizer = Pipeline(
    steps=[
        ('vectorizer', vectorizer)
    ]
)

estimator = SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    max_iter=5_000,
    random_state=42
)

estimator = MultiOutputClassifier(
    estimator=estimator
)

pipeline = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator)
    ]
)

params = {
    'standardizer__vectorizer__norm': ['categorical', [None, 'l1', 'l2']],
    'standardizer__vectorizer__sublinear_tf': ['categorical', [True, False]],
    'standardizer__vectorizer__max_features': ['int', {'low': 50_000,
                                                       'high': 250_000,
                                                       'step': 50_000}],
    'standardizer__vectorizer__min_df': ['int', {'low': 2,
                                                 'high': 20,
                                                 'step': 2}],
    'standardizer__vectorizer__max_df': ['float', {'low': 0.7,
                                                   'high': 1.0,
                                                   'step': 0.05}],
    'estimator__estimator__alpha': ['float', {'low': 0.1,
                                              'high': 1.0,
                                              'step': 0.05}],
    'estimator__estimator__class_weight': ['categorical', [None, 'balanced']],
    'estimator__estimator__l1_ratio': ['float', {'low': 0.1,
                                                 'high': 1.0,
                                                 'step': 0.05}]
}

scoring = make_scorer(
    score_func=f1_score,
    average='weighted',
    zero_division=0.0
)

model = Model(
    pipeline=pipeline,
    name='SGDClassifier',
    params=params,
    metric=metric,
    scoring=scoring,
    cv=2
)
