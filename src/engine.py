"""Load the selected news corpus and initialize the notebook's NLP components."""

from pathlib import Path
from collections import Counter

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .analysis import NewsClassifier, SentimentAnalyzer, TopicModeler, extract_entities, get_nlp
from .discovery import SemanticSearch, generate_insights, summarize
from .multilingual import LANG_NAMES, detect_language, translate
from .preprocessing import preprocess_text

ROOT = Path(__file__).resolve().parents[1]
BBC_DATA = ROOT / "data" / "BBC News Train.csv"
DEMO_DATA = ROOT / "data" / "wikinews-demo.csv"
NLTK_DATA = ROOT / ".nltk_data"
RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet.zip",
    "vader_lexicon": "sentiment/vader_lexicon.zip",
}


def prepare_resources():
    path = str(NLTK_DATA)
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)
    for package, resource in RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, download_dir=path, quiet=True):
                raise RuntimeError(f"Could not install NLTK resource {package}.")
    get_nlp()


def load_dataset(path):
    frame = pd.read_csv(path).rename(columns={
        "ArticleId": "article_id", "Text": "content", "Category": "category",
        "Title": "title", "SourceURL": "source_url",
    })
    frame = frame.dropna(subset=["content", "category"])
    frame = frame[frame["content"].str.strip().ne("")]
    return frame.drop_duplicates(subset="content").reset_index(drop=True)


class NewsBot2:
    def __init__(self):
        prepare_resources()
        self.is_demo = not BBC_DATA.is_file()
        self.corpus_name = "Wikinews" if self.is_demo else "BBC"
        self.df = load_dataset(DEMO_DATA if self.is_demo else BBC_DATA)
        self.df["processed"] = self.df["content"].map(preprocess_text)
        train, test = train_test_split(
            self.df, test_size=0.2, random_state=42, stratify=self.df["category"],
        )
        with threadpool_limits(limits=2):
            self.classifier = NewsClassifier().fit(train["processed"], train["category"], tune=True)
            self.test_accuracy, _ = self.classifier.evaluate(test["processed"], test["category"])
            self.topic_models = {
                method: TopicModeler(method=method).fit(self.df["processed"])
                for method in ("nmf", "lda")
            }
        self.searcher = SemanticSearch().fit(self.df["content"])
        self.sentiment = SentimentAnalyzer()
        self.sentiment_frame = self.sentiment.analyze_df(self.df)
        self.sentiment_scores = self.sentiment_frame["compound"]
        self.sentiment_labels = self.sentiment_frame["label"]

    def classify(self, text):
        processed = preprocess_text(text)
        if not processed or not self.classifier.model["tfidf"].transform([processed]).nnz:
            raise ValueError("Enter English news text with recognizable vocabulary.")
        return self.classifier.predict_top_k(processed, k=5)

    def analyze_article(self, text):
        predictions = self.classify(text)
        category, confidence = predictions[0]
        sentiment = self.sentiment.analyze(text)
        processed = preprocess_text(text)
        topics = {}
        for method, model in self.topic_models.items():
            if model.vectorizer.transform([processed]).nnz:
                topics[method] = model.transform([processed])[0]
        return {
            "category": category, "confidence": confidence, "predictions": predictions,
            "sentiment": sentiment["label"], "scores": sentiment,
            "entities": Counter(extract_entities(text)), "topics": topics,
        }

    def analyze_language(self, text):
        language, confidence = detect_language(text)
        english, ok = (text, True) if language == "en" else translate(
            text, "en", language if language != "unknown" else "auto",
        )
        result = {
            "Language": LANG_NAMES.get(language, language), "Detection": confidence,
            "English": english if ok else "", "Status": "English input" if language == "en"
            else "Translated" if ok else "Translation unavailable",
        }
        if ok:
            result["Tone"] = self.sentiment.analyze(english)["label"]
            try:
                result["Category"], result["Confidence"] = self.classify(english)[0]
            except ValueError:
                result["Status"] += " · insufficient classification vocabulary"
        return result

    def search(self, query, top_k=5):
        return self.searcher.search(query, top_k)

    def summarize(self, text, n=3):
        return summarize(text, n)

    def topics(self, n=10, method="nmf"):
        return self.topic_models[method].top_words(n)

    def insights(self):
        return generate_insights(self.df, self.sentiment_frame, self.topics())

    def report(self):
        lines = [
            "# NewsBot 2.0 Report", f"Corpus: {self.corpus_name}", f"Articles: {len(self.df)}",
            f"Holdout classification accuracy: {self.test_accuracy:.4f}", "", "## Topics",
        ]
        lines.extend(f"- {topic}: {', '.join(words)}" for topic, words in self.topics().items())
        return "\n".join([*lines, "", "## Insights", *(f"- {line}" for line in self.insights())])
