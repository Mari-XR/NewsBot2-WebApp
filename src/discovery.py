from .preprocessing import preprocess_text, split_sentences

"""Module B: extractive summarization (TF-IDF scoring + TextRank)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def summarize(text, n_sentences=3, method="textrank"):
    sents = split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= n_sentences:
        return " ".join(sents)
    if not TfidfVectorizer(stop_words="english").build_analyzer()(" ".join(sents)):
        return " ".join(sents[:n_sentences])
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    if method == "tfidf":
        scores = np.asarray(tfidf.sum(axis=1)).ravel() / (np.asarray((tfidf > 0).sum(axis=1)).ravel() + 1)
    else:  # TextRank: PageRank over the sentence-similarity graph
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0)
        row = sim.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        m = sim / row
        scores = np.ones(len(sents)) / len(sents)
        for _ in range(50):
            scores = 0.15 / len(sents) + 0.85 * m.T.dot(scores)
    top = sorted(np.argsort(scores)[::-1][:n_sentences])  # keep original order
    return " ".join(sents[i] for i in top)


def lead_baseline(text, n_sentences=3):
    return " ".join(split_sentences(text)[:n_sentences])

"""Module B: semantic embeddings + search with TF-IDF cosine similarity
(optional: sentence-transformers embeddings if installed)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    def __init__(self, use_embeddings=False):
        self.use_embeddings = use_embeddings
        self.encoder = None

    def fit(self, texts):
        self.texts = list(texts)
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.matrix = self.encoder.encode(self.texts, show_progress_bar=False)
                return self
            except Exception as e:
                print("Embeddings not available, using TF-IDF instead:", e)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([preprocess_text(t) for t in self.texts])
        return self

    def search(self, query, top_k=5):
        if self.encoder is not None:
            q = self.encoder.encode([query])
        else:
            q = self.vectorizer.transform([preprocess_text(query)])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]

    def similar_to(self, doc_index, top_k=5):
        sims = cosine_similarity(self.matrix[doc_index:doc_index + 1] if self.encoder is None
                                 else self.matrix[[doc_index]], self.matrix).ravel()
        sims[doc_index] = -1
        idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idx]

    def expand_query(self, query, n_terms=3):
        """Query expansion: add words that often appear in the top results (pseudo-relevance feedback)."""
        if self.encoder is not None:
            return query
        hits = self.search(query, top_k=5)
        if not hits:
            return query
        names = self.vectorizer.get_feature_names_out()
        centroid = np.asarray(self.matrix[[i for i, _ in hits]].mean(axis=0)).ravel()
        q_words = set(preprocess_text(query).split())
        extra = [names[i] for i in centroid.argsort()[::-1] if " " not in names[i] and names[i] not in q_words][:n_terms]
        return query + " " + " ".join(extra)

"""Module B: content enhancement and insight generation (template-based, no LLM needed)."""
from collections import Counter


def enhance_article(text, summary, category, confidence, sentiment, entities, related):
    """Build an 'enhanced' article card: summary + context (entities, related articles)."""
    people = [e for e, l in entities if l == "PERSON"]
    orgs = [e for e, l in entities if l == "ORG"]
    places = [e for e, l in entities if l == "GPE"]
    top = lambda xs: ", ".join(w for w, _ in Counter(xs).most_common(3)) or "none found"
    return (f"Category: {category} ({confidence:.0%} confident) | Tone: {sentiment}\n"
            f"Summary: {summary}\n"
            f"Key people: {top(people)} | Organizations: {top(orgs)} | Places: {top(places)}\n"
            f"Related articles: {', '.join(str(i) for i, _ in related) or 'none'}")


def generate_insights(df, sent_df=None, topic_words=None):
    """Turn numbers into short plain-English findings."""
    out = []
    counts = df["category"].value_counts()
    out.append(f"The biggest category is {counts.index[0]} with {counts.iloc[0]} articles; "
               f"the smallest is {counts.index[-1]} with {counts.iloc[-1]}.")
    if sent_df is not None:
        m = sent_df.groupby("category")["compound"].mean().sort_values()
        out.append(f"The most negative category is {m.index[0]} (avg {m.iloc[0]:.2f}); "
                   f"the most positive is {m.index[-1]} (avg {m.iloc[-1]:.2f}).")
    if topic_words:
        for k, words in topic_words.items():
            out.append(f"{k} is about: {', '.join(words[:5])}.")
    return out
