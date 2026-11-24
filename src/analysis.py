TFIDF_PARAMS = dict(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)

"""Module A: enhanced classification with confidence scores."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score


def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", model),
    ])


class NewsClassifier:
    """TF-IDF + Logistic Regression, tuned with a small grid search."""

    def __init__(self):
        self.model = None
        self.results = {}

    def compare_models(self, X_train, y_train, X_test, y_test):
        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Linear SVM": LinearSVC(),
        }
        for name, m in models.items():
            pipe = make_pipeline(m).fit(X_train, y_train)
            pred = pipe.predict(X_test)
            self.results[name] = {
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
            }
        return self.results

    def fit(self, X_train, y_train, tune=True):
        pipe = make_pipeline(LogisticRegression(max_iter=2000))
        if tune:
            grid = GridSearchCV(pipe, {"clf__C": [1, 5, 10, 20]}, cv=3, n_jobs=1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
        else:
            self.model = pipe.fit(X_train, y_train)
            self.best_params = {}
        return self

    def evaluate(self, X_test, y_test):
        pred = self.model.predict(X_test)
        return accuracy_score(y_test, pred), classification_report(y_test, pred)

    def predict(self, texts):
        return self.model.predict(texts)

    def predict_top_k(self, text, k=2):
        """Multi-level output: the top-k categories with probabilities."""
        probs = self.model.predict_proba([text])[0]
        order = np.argsort(probs)[::-1][:k]
        return [(self.model.classes_[i], float(probs[i])) for i in order]

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)
        return self

    def predict_with_confidence(self, text):
        probs = self.model.predict_proba([text])[0]
        i = int(np.argmax(probs))
        return self.model.classes_[i], float(probs[i])

"""Module A: topic discovery with LDA and NMF."""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


class TopicModeler:
    def __init__(self, n_topics=5, method="nmf", random_state=42):
        self.n_topics = n_topics
        self.method = method
        if method == "lda":
            self.vectorizer = CountVectorizer(max_features=3000, min_df=5, max_df=0.7)
            self.model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method="batch")
        else:
            self.vectorizer = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.7)
            self.model = NMF(n_components=n_topics, random_state=random_state, init="nndsvd", max_iter=400)

    def fit(self, processed_texts):
        self.dtm = self.vectorizer.fit_transform(processed_texts)
        self.doc_topics = self.model.fit_transform(self.dtm)
        return self

    def fit_transform(self, documents):
        """Train topic model and return the document-topic matrix."""
        return self.fit(documents).doc_topics

    def get_topic_words(self, topic_id, n_words=10):
        return self.top_words(n_words)[f"Topic {topic_id}"]

    def topic_evolution(self, periods):
        """Share of each dominant topic per period (topic trend over 'time')."""
        return pd.crosstab(pd.Series(list(periods), name="period"),
                           pd.Series(self.dominant_topic(), name="topic"), normalize="index")

    def cluster_articles(self, n_clusters=5, random_state=42):
        """Group similar articles using their topic distributions (KMeans)."""
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(self.doc_topics)

    def top_words(self, n=10):
        words = self.vectorizer.get_feature_names_out()
        return {f"Topic {k}": [words[i] for i in comp.argsort()[::-1][:n]]
                for k, comp in enumerate(self.model.components_)}

    def dominant_topic(self):
        return self.doc_topics.argmax(axis=1)

    def topic_vs_category(self, categories):
        return pd.crosstab(pd.Series(self.dominant_topic(), name="topic"),
                           pd.Series(list(categories), name="category"))

    def transform(self, processed_texts):
        return self.model.transform(self.vectorizer.transform(processed_texts))

from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """VADER sentiment (same as midterm) with a simple label."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        s = self.sia.polarity_scores(str(text))
        if s["compound"] >= 0.05:
            s["label"] = "positive"
        elif s["compound"] <= -0.05:
            s["label"] = "negative"
        else:
            s["label"] = "neutral"
        return s

    def analyze_df(self, df, text_col="content"):
        scores = df[text_col].map(self.analyze)
        out = df[["article_id", "category"]].copy()
        out["compound"] = scores.map(lambda s: s["compound"])
        out["label"] = scores.map(lambda s: s["label"])
        return out

    def sentiment_evolution(self, sent_df, time_col="period"):
        """Average sentiment per time period and category.
        NOTE: the BBC dataset has no dates, so we use article_id order split into
        periods as a stand-in for time. With a dated dataset, pass a real date column."""
        return sent_df.groupby([time_col, "category"])["compound"].mean().unstack()

from functools import lru_cache

import spacy


@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("en_core_web_sm")


def extract_entities(text):
    """Return a list of (entity_text, label)."""
    doc = get_nlp()(str(text))
    return [(e.text.strip(), e.label_) for e in doc.ents]
