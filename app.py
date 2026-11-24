"""Streamlit entry point for Purple Berries' NewsBot 2.0."""

import base64
import logging
import re

import pandas as pd
import streamlit as st

from src.chat import QueryProcessor
from src.discovery import enhance_article
from src.engine import ROOT, NewsBot2
from src.multilingual import LANG_NAMES, translate

TITLE = "NewsBot Intelligence System 2.0"
PAGES = ("Analyze News", "Summarize & Search", "Multilingual Analysis", "Chat with NewsBot")
DESCRIPTIONS = {
    PAGES[0]: "Classify a news article and inspect its tone, named entities and topics.",
    PAGES[1]: "Summarize a story or discover related articles in the news archive.",
    PAGES[2]: "Detect language, translate articles and compare the English-model results.",
    PAGES[3]: "Explore news articles with follow-up questions and article references.",
}
ARTICLE_LIMIT = 12000
TRANSLATION_LIMIT = 4500


@st.cache_resource(show_spinner="Preparing the news corpus and NLP models…")
def get_bot():
    return NewsBot2()


@st.cache_data
def theme():
    image = (ROOT / "assets" / "newsbot_background.png").read_bytes()
    background = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    return (ROOT / "assets" / "theme.css").read_text().replace("BACKGROUND_IMAGE", background)


def article_input(label, key, maximum=ARTICLE_LIMIT):
    return st.text_area(label, key=key, max_chars=maximum, height=180).strip()


def render_hits(bot, hits):
    if not hits:
        st.info("No matching articles found.")
    for index, score in hits:
        article = bot.df.loc[index]
        with st.expander(f"Article {index} · {article.category.title()} · {score:.1%} similarity"):
            if bot.is_demo:
                st.caption(article.title)
                st.link_button("Original and contributor history", article.source_url)
            else:
                st.caption(f"BBC ArticleId {article.article_id}")
            st.text(article.content)


def analyze_page(bot):
    st.header(PAGES[0])
    with st.form("analyze"):
        text = article_input("English news article", "analyze_text")
        submitted = st.form_submit_button("Analyze article", type="primary")
    if not submitted:
        return
    try:
        result = bot.analyze_article(text)
    except ValueError as error:
        st.warning(str(error))
        return
    columns = st.columns(3)
    columns[0].metric("Category", result["category"].title())
    columns[1].metric("Classification confidence", f"{result['confidence']:.1%}")
    columns[2].metric("VADER tone", result["sentiment"].title(), f"{result['scores']['compound']:+.3f} compound",
                      delta_color="off")
    predictions, entities, topics = st.tabs(["Category probabilities", "Named entities", "Topics"])
    with predictions:
        frame = pd.DataFrame(result["predictions"], columns=["Category", "Probability"])
        st.bar_chart(frame, x="Category", y="Probability", horizontal=True)
        st.dataframe(frame, hide_index=True, column_config={
            "Probability": st.column_config.NumberColumn(format="percent"),
        })
        st.caption("VADER compound ranges from −1 to +1.")
    with entities:
        records = [(name, label, count) for (name, label), count in result["entities"].most_common()]
        if records:
            st.dataframe(pd.DataFrame(records, columns=["Entity", "spaCy label", "Mentions"]), hide_index=True)
            st.caption("PERSON identifies people, ORG organizations, and GPE or LOC places.")
        else:
            st.info("spaCy found no named entities.")
    with topics:
        if not result["topics"]:
            st.info("The topic models found no matching vocabulary.")
        for method, scores in result["topics"].items():
            st.subheader(method.upper())
            words = bot.topics(6, method)
            rows = [(topic, float(score), ", ".join(terms)) for (topic, terms), score in zip(words.items(), scores)]
            st.dataframe(pd.DataFrame(rows, columns=["Topic", "Weight", "Top terms"]).sort_values(
                "Weight", ascending=False), hide_index=True)
        if result["topics"]:
            st.caption("LDA and NMF return weights on different scales.")


def discovery_page(bot):
    st.header(PAGES[1])
    summary_tab, search_tab, insights_tab = st.tabs(["Article summary", f"{bot.corpus_name} search", "Corpus insights"])
    with summary_tab:
        with st.form("summary"):
            text = article_input("English news article", "summary_text")
            count = st.slider("Summary sentences", 1, 6, 3)
            summarize = st.form_submit_button("Generate summary", type="primary")
        if summarize:
            summary = bot.summarize(text, count)
            if not summary:
                st.warning("Enter an article with at least one complete sentence of four words or more.")
            else:
                st.subheader("TextRank summary")
                st.text(summary)
                with st.expander("Original article"):
                    st.text(text)
                hits = bot.search(summary)
                st.subheader(f"Related {bot.corpus_name} articles")
                render_hits(bot, hits)
                try:
                    analysis = bot.analyze_article(text)
                except ValueError:
                    analysis = None
                if analysis:
                    with st.expander("Article brief"):
                        st.text(enhance_article(text, summary, analysis["category"], analysis["confidence"],
                                                analysis["sentiment"], list(analysis["entities"].elements()), hits))
    with search_tab:
        with st.form("search"):
            query = st.text_input(f"Search the {bot.corpus_name} archive", max_chars=350).strip()
            expand = st.checkbox("Expand query with terms from the top matches")
            search = st.form_submit_button("Search articles", type="primary")
        if search:
            if not query:
                st.warning("Enter a search term.")
            else:
                expanded = bot.searcher.expand_query(query).strip() if expand else query
                if expanded != query:
                    st.text(f"Expanded query · {expanded}")
                render_hits(bot, bot.search(expanded, top_k=8))
    with insights_tab:
        for line in bot.insights():
            st.text(line)
        st.download_button("Download corpus report", bot.report(), "newsbot_report.md", "text/markdown")


def multilingual_page(bot):
    st.header(PAGES[2])
    with st.form("multilingual"):
        text = article_input("Article text", "multilingual_text", TRANSLATION_LIMIT)
        comparison = article_input("Another version of the story (optional)", "compare_text", TRANSLATION_LIMIT)
        target = st.selectbox("Additional translation", [None, *LANG_NAMES],
                              format_func=lambda code: LANG_NAMES.get(code, "None"))
        st.caption("For translation, article text is sent to Google Translate or MyMemory.")
        analyze = st.form_submit_button("Analyze languages", type="primary")
    if not analyze:
        return
    if not text:
        st.warning("Enter the first article to analyze.")
        return
    texts = [text, comparison] if comparison else [text]
    with st.spinner("Detecting languages and translating as needed…"):
        results = [bot.analyze_language(item) for item in texts]
        target_text, target_ok = "", False
        if target and results[0]["English"]:
            target_text, target_ok = translate(results[0]["English"], target, "en")
    frame = pd.DataFrame(results).drop(columns="English")
    st.dataframe(frame, hide_index=True, column_config={
        key: st.column_config.NumberColumn(format="percent") for key in ["Detection", "Confidence"]
    })
    for pos, (original, result) in enumerate(zip(texts, results), 1):
        with st.expander(f"Article {pos} · {result['Language']}", expanded=True):
            st.markdown("**Original**")
            st.text(original)
            if result["Status"].startswith("Translated"):
                st.markdown("**English translation**")
                st.text(result["English"])
            elif not result["English"]:
                st.warning("English translation is unavailable. English-model analysis could not run.")
    if target:
        if target_ok:
            st.subheader(f"Article 1 · {LANG_NAMES[target]}")
            st.text(target_text)
        else:
            st.warning(f"The {LANG_NAMES[target]} translation is unavailable.")


def chat_sources(bot, response):
    if not bot.is_demo:
        return
    matches = re.findall(
        r"^\[(\d+)\]|^(?:Summary of article|Translated summary of article|"
        r"Translation unavailable\. Original summary of article) (\d+)",
        response, re.M,
    )
    indexes = dict.fromkeys(int(list_id or article_id) for list_id, article_id in matches)
    indexes = [index for index in indexes if index in bot.df.index]
    if indexes:
        with st.expander("Wikinews sources and contributors"):
            for index in indexes:
                article = bot.df.loc[index]
                st.link_button(f"Article {index} · {article.title}", article.source_url)


def chat_page(bot):
    st.header(PAGES[3])
    st.caption("Try “Show me positive tech articles”, “More”, or “Summarize article 5”. "
               f"Article numbers refer to the {bot.corpus_name} archive.")
    if st.button("Clear conversation"):
        st.session_state.pop("news_chat", None)
        st.session_state.pop("news_messages", None)
    if "news_chat" not in st.session_state:
        st.session_state.news_chat = QueryProcessor(bot)
        st.session_state.news_messages = []
    messages = st.session_state.news_messages
    for role, message in messages:
        with st.chat_message(role):
            st.text(message)
            if role == "assistant":
                chat_sources(bot, message)
    if prompt := st.chat_input(f"Ask about the {bot.corpus_name} archive", max_chars=4500):
        prompt = prompt.strip()
        if not prompt:
            return
        with st.chat_message("user"):
            st.text(prompt)
        with st.chat_message("assistant"):
            with st.spinner(f"Checking the {bot.corpus_name} archive…"):
                response = st.session_state.news_chat.process(prompt)
            st.text(response)
            chat_sources(bot, response)
        messages.extend([("user", prompt), ("assistant", response)])


def main():
    st.set_page_config(page_title=TITLE, page_icon="🗞️", layout="wide")
    st.html(f"<style>{theme()}</style>")
    st.sidebar.title("NewsBot 2.0")
    page = st.sidebar.radio("Workspace", PAGES)
    st.sidebar.divider()
    st.sidebar.markdown("**Purple Berries**  \nITAI 2373 · Natural Language Processing")
    st.sidebar.markdown("Marisabel Morales  \nKhalida Bestani  \nSadea Saeed")
    st.html(f'<section class="hero"><p class="eyebrow">PURPLE BERRIES · ITAI 2373</p>'
            f'<h1>{TITLE}</h1><p>{DESCRIPTIONS[page]}</p></section>')
    try:
        bot = get_bot()
    except Exception:
        logging.exception("NewsBot initialization failed")
        st.error("NewsBot could not initialize. Check the dataset, dependencies and NLTK downloads, then reload.")
        return
    st.sidebar.caption(f"{len(bot.df):,} {bot.corpus_name} articles · {bot.df.category.nunique()} categories")
    if bot.is_demo:
        st.sidebar.markdown(
            "Public demo · adapted excerpts by Wikinews contributors · "
            "[CC BY 2.5](https://creativecommons.org/licenses/by/2.5/)"
        )
    pages = dict(zip(PAGES, (analyze_page, discovery_page, multilingual_page, chat_page)))
    pages[page](bot)


if __name__ == "__main__":
    main()
