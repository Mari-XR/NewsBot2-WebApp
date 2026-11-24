from .multilingual import translate

"""Module D: rule-based intent classification (keywords + regex)."""
import re

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
SYNONYMS = {"sports": "sport", "technology": "tech", "political": "politics",
            "finance": "business", "economy": "business", "movies": "entertainment",
            "music": "entertainment", "film": "entertainment"}


def find_category(text):
    words = re.findall(r"[a-z]+", text.lower())
    for w in words:
        if w in CATEGORIES:
            return w
        if w in SYNONYMS:
            return SYNONYMS[w]
    return None


def find_sentiment_filter(text):
    q = text.lower()
    if re.search(r"\b(positive|good|happy)\b", q):
        return "positive"
    if re.search(r"\b(negative|bad|sad)\b", q):
        return "negative"
    return None


def detect_intent(query):
    q = query.lower().strip()
    num = re.search(r"\b(\d+)\b", q)
    if re.search(r"\b(hi|hello|hey)\b", q):
        return "greeting", {}
    if re.search(r"\b(help|what can you do)\b", q):
        return "help", {}
    if "summar" in q and num:
        return "summarize", {"index": int(num.group(1))}
    if "sentiment" in q or "mood" in q or "feel" in q:
        return "sentiment", {"category": find_category(q), "index": int(num.group(1)) if num else None}
    if "topic" in q:
        return "topics", {}
    if re.search(r"\b(translate)\b", q):
        m = re.search(r"translate\s+(?:article\s+)?(\d+)\s+(?:to|into)\s+(\w+)", q)
        if m:
            return "translate", {"index": int(m.group(1)), "lang": m.group(2)}
    if re.search(r"\b(classify|categor)", q):
        text = re.sub(r"^.*?(classify|categorize)\s*:?\s*", "", query, flags=re.I)
        return "classify", {"text": text}
    m = re.search(r"\b(?:who is|tell me about|news about|mentions of)\s+(.+)$", q)
    if m and not re.search(r"\barticles?\b", q):
        return "entity", {"name": m.group(1).strip(" ?.")}
    if re.search(r"\b(more|another|next|show more)\b", q) and len(q.split()) <= 4:
        return "follow_up", {}
    if re.search(r"\b(report|export)\b", q):
        return "export", {}
    if re.search(r"\b(find|search|about|related)\b", q):
        m = re.search(r"(?:about|for|on|related to)\s+(.+)$", q)
        return "search", {"query": m.group(1) if m else q}
    if re.search(r"\b(show|list|give)\b", q) and find_category(q):
        return "show_category", {"category": find_category(q), "sentiment": find_sentiment_filter(q)}
    if find_category(q):
        return "show_category", {"category": find_category(q), "sentiment": find_sentiment_filter(q)}
    return "unknown", {}


def split_multi_part(query):
    """Handle multi-part questions like 'show tech articles and summarize article 3'."""
    parts = re.split(r"\s+(?:and then|and also|then|;)\s+|\s+and\s+(?=(?:show|summar|find|what|translate|classify|list))", query, flags=re.I)
    return [p for p in parts if p and p.strip()]

"""Module D: turns an intent + slots into a friendly answer using the NewsBot components."""

HELP_TEXT = """I can help with:
  - "show positive sports articles"
  - "summarize article 5"
  - "what's the sentiment of business news"
  - "find articles about elections"
  - "who is tony blair"           (entity lookup)
  - "what are the main topics"
  - "translate article 3 to spanish"
  - "classify: <some news text>"
  - "more" / "next"               (follow-up on the last list)
You can also combine: "show tech articles and summarize article 3"."""

LANG_CODES = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
              "portuguese": "pt", "arabic": "ar", "chinese": "zh-CN"}


class ResponseGenerator:
    def __init__(self, bot):
        self.bot = bot

    def _list(self, sub, title):
        lines = [f"[{i}] {t[:85]}..." for i, t in zip(sub.index, sub["content"])]
        return title + "\n\n" + "\n\n".join(lines)

    def respond(self, intent, s, ctx):
        b, df = self.bot, self.bot.df
        if intent == "greeting":
            return "Hi! I'm NewsBot 2.0. Type 'help' to see what I can do."
        if intent == "help":
            return HELP_TEXT
        if intent == "show_category":
            sub = df[df["category"] == s["category"]]
            if s.get("sentiment"):
                sub = sub[b.sentiment_labels.loc[sub.index] == s["sentiment"]]
            ctx["last_results"], ctx["offset"] = list(sub.index), 5
            ctx["last_category"] = s["category"]
            ctx["sentiment_scope"] = list(sub.index)
            ctx.pop("last_article", None)
            mood = f"{s['sentiment']} " if s.get("sentiment") else ""
            return self._list(sub.head(5), f"Showing {min(5, len(sub))} of {len(sub)} {mood}{s['category']} articles:")
        if intent == "follow_up":
            res, off = ctx.get("last_results", []), ctx.get("offset", 0)
            if not res:
                return "There is nothing to continue yet. Try 'show tech articles' first."
            ctx["offset"] = off + 5
            nxt = res[off:off + 5]
            return self._list(df.loc[nxt], "Next results:") if nxt else "No more results."
        if intent == "summarize":
            i = s["index"]
            if i not in df.index:
                return f"Sorry, article {i} does not exist (0-{len(df) - 1})."
            ctx["last_article"] = i
            return f"Summary of article {i} ({df.loc[i, 'category']}):\n" + b.summarize(df.loc[i, "content"])
        if intent == "sentiment":
            i = s.get("index")
            if i is None and not s.get("category"):
                i = ctx.get("last_article")
            if i is not None:
                if i not in df.index:
                    return f"Article {i} does not exist (0-{len(df) - 1})."
                result = b.sentiment.analyze(df.loc[i, "content"])
                return f"Article {i} is {result['label']} (compound={result['compound']:.2f})."
            cat = s.get("category")
            if not cat and "sentiment_scope" in ctx:
                scores = b.sentiment_scores.loc[ctx["sentiment_scope"]]
                cat = "the previous results"
            else:
                scores = b.sentiment_scores[df["category"].eq(cat)] if cat else b.sentiment_scores
            if scores.empty:
                return "No articles in that group."
            return (f"Sentiment of {cat or 'all news'}: average compound {scores.mean():.2f}, "
                    f"{(scores >= 0.05).mean() * 100:.0f}% positive articles.")
        if intent == "search":
            q = b.searcher.expand_query(s["query"])
            hits = b.search(q, top_k=len(df))
            ctx["last_results"], ctx["offset"] = [i for i, _ in hits], 5
            ctx["sentiment_scope"] = ctx["last_results"]
            ctx.pop("last_article", None)
            if not hits:
                return "No matching articles found."
            return f"Top articles about '{s['query']}' (expanded query: '{q}'):\n" + "\n".join(
                f"[{i}] ({df.loc[i, 'category']}, score {sc:.2f}) {df.loc[i, 'content'][:75]}..." for i, sc in hits[:5])
        if intent == "entity":
            name = s["name"].lower()
            mask = df["content"].str.lower().str.contains(name, regex=False)
            if not mask.any():
                return f"I could not find '{s['name']}' in the articles."
            cats = df[mask]["category"].value_counts().to_dict()
            mood = b.sentiment_scores[mask].mean()
            ctx["last_results"], ctx["offset"] = list(df[mask].index), 0
            ctx["sentiment_scope"] = ctx["last_results"]
            ctx.pop("last_article", None)
            return (f"'{s['name']}' appears in {int(mask.sum())} articles {cats}. "
                    f"Average tone of those articles: {mood:.2f}. Say 'more' to list them.")
        if intent == "topics":
            return "Main topics:\n" + "\n".join(f"{k}: {', '.join(v[:8])}" for k, v in b.topics().items())
        if intent == "translate":
            i = s["index"]
            if i not in df.index:
                return f"Sorry, article {i} does not exist."
            code = LANG_CODES.get(s["lang"], s["lang"])
            original = b.summarize(df.loc[i, "content"], 2)
            ctx["last_article"] = i
            text, ok = translate(original, target=code, source="en")
            if ok:
                return f"Translated summary of article {i}:\n\n{text}"
            return f"Translation unavailable. Original summary of article {i}:\n\n{original}"
        if intent == "classify":
            if not s["text"].strip():
                return "Please give me some text, e.g. 'classify: the team won the cup final'."
            try:
                r = b.analyze_article(s["text"])
            except ValueError as error:
                return str(error)
            return f"Category: {r['category']} ({r['confidence']:.0%}), sentiment: {r['sentiment']}."
        if intent == "export":
            return "Use Download corpus report on the Summarize & Search page."
        return "Sorry, I did not understand. Type 'help' for examples."

"""Module D: main entry for natural-language questions. Keeps conversation state (context)."""
import logging

log = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(self, bot):
        self.responder = ResponseGenerator(bot)
        self.context = {}      # remembers last results, last category, last article
        self.history = []

    def process(self, query):
        answers = []
        for part in split_multi_part(query):
            intent, slots = detect_intent(part)
            try:
                ans = self.responder.respond(intent, slots, self.context)
            except Exception as e:  # Keep the conversation usable after unexpected errors.
                log.error("Chat failed for intent %s (%s)", intent, type(e).__name__)
                ans = f"Something went wrong ({type(e).__name__}). Please try another question."
            self.history.append({"query": part, "intent": intent, "answer": ans})
            answers.append(ans)
        return "\n\n".join(answers)
