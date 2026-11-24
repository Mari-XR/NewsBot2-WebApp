"""Run with python -m unittest discover -s tests -v."""
import os
import re
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from src.chat import QueryProcessor
from src.engine import NewsBot2, ROOT
from src.multilingual import detect_language, translate

SPANISH = "El gobierno anunció nuevas medidas para ayudar a las familias que necesitan apoyo económico."
ENGLISH = "The government announced new support measures for local families."
ARTICLE = (
    "Apple announced a new iPhone at its headquarters in California. "
    "The new iPhone offers better battery life and a faster processor. "
    "Apple said the iPhone will reach stores next week. "
    "The smartphone launch brought good news for technology investors."
)


class NewsBotSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = NewsBot2()

    def test_corpus_and_classifier(self):
        self.assertEqual(len(self.bot.df), 300 if self.bot.is_demo else 1440)
        if self.bot.is_demo:
            self.assertTrue(self.bot.df.source_url.str.startswith(
                "https://en.wikinews.org/wiki/Special:PermanentLink/"
            ).all())
        else:
            self.assertEqual(self.bot.classifier.best_params, {"clf__C": 10})
            self.assertAlmostEqual(self.bot.test_accuracy, 278 / 288)
        self.assertEqual(self.bot.df.category.nunique(), 5)
        self.assertFalse(self.bot.df.content.duplicated().any())
        result = self.bot.analyze_article(ARTICLE)
        self.assertIn(result["category"], self.bot.df.category.unique())
        if not self.bot.is_demo:
            self.assertEqual(result["category"], "tech")
        self.assertAlmostEqual(sum(score for _, score in result["predictions"]), 1.0)
        self.assertIn(("Apple", "ORG"), result["entities"])
        self.assertIn(("California", "GPE"), result["entities"])
        self.assertEqual(set(result["topics"]), {"lda", "nmf"})
        self.assertEqual(self.bot.sentiment.analyze("This is excellent, I love it!")["label"], "positive")

    def test_discovery(self):
        summary = self.bot.summarize(ARTICLE, 2)
        self.assertTrue(summary and len(summary) < len(ARTICLE))
        hits = self.bot.search("elections and voters")
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(score > 0 and index in self.bot.df.index for index, score in hits))
        self.assertNotEqual(self.bot.searcher.expand_query("elections"), "elections")
        self.assertIn(f"Corpus: {self.bot.corpus_name}", self.bot.report())

    def test_context(self):
        chat = QueryProcessor(self.bot)
        refs = lambda message: re.findall(r"^\[(\d+)\]", message, re.M)
        first = refs(chat.process("Show me positive tech articles"))
        more = refs(chat.process("More"))
        self.assertEqual((len(first), len(more)), (5, 5))
        self.assertFalse(set(first) & set(more))
        self.assertIn("100% positive", chat.process("What about the sentiment?"))
        self.assertIn("Summary of article 5", chat.process("Summarize article 5"))
        self.assertIn("Article 5 is", chat.process("What about the sentiment?"))
        self.assertEqual(len(refs(chat.process("Find articles about elections"))), 5)
        self.assertEqual(len(refs(chat.process("More"))), 5)
        self.assertIn("does not exist", chat.process("Summarize article 9999"))
        other = QueryProcessor(self.bot)
        self.assertIn("nothing to continue", other.process("More"))

    def test_detection(self):
        examples = {
            "en": ENGLISH, "es": SPANISH,
            "fr": "Le gouvernement a annoncé de nouvelles mesures pour soutenir les familles locales.",
            "de": "Die Regierung kündigte neue Maßnahmen zur Unterstützung lokaler Familien an.",
            "it": "Il governo ha annunciato nuove misure per aiutare le famiglie della zona.",
        }
        for language, sample in examples.items():
            with self.subTest(language=language):
                self.assertEqual(detect_language(sample)[0], language)
        self.assertEqual(self.bot.analyze_language(ENGLISH)["Status"], "English input")

    def test_translation_failure_and_fallback(self):
        with patch("src.multilingual.google", side_effect=requests.Timeout), patch(
            "src.multilingual.mymemory", return_value=ENGLISH
        ) as backup:
            self.assertEqual(translate(SPANISH, "en", "es"), (ENGLISH, True))
            backup.assert_called_once()
        with patch("src.multilingual.google", side_effect=lambda session, text, source, target: text), patch(
            "src.multilingual.mymemory", side_effect=requests.Timeout
        ):
            result = self.bot.analyze_language(SPANISH)
            self.assertEqual(result["Status"], "Translation unavailable")
            self.assertNotIn("Category", result)
            chat = QueryProcessor(self.bot)
            self.assertTrue(chat.process("Translate article 3 to Spanish").startswith("Translation unavailable"))

    @unittest.skipUnless(os.getenv("NEWSBOT_TEST_TRANSLATION") == "1", "Live provider test requires internet")
    def test_live_translation(self):
        for sample, target, source in [(SPANISH, "en", "es"), (ENGLISH, "es", "en")]:
            result, success = translate(sample, target, source)
            self.assertTrue(success)
            self.assertNotEqual(result, sample)
            self.assertEqual(detect_language(result)[0], target)
        self.assertIn("Category", self.bot.analyze_language(SPANISH))
        self.assertIn("Translated summary of article 3",
                      QueryProcessor(self.bot).process("Translate article 3 to Spanish"))

    def test_invalid_input(self):
        for text in ("", "   ", "123!", "qzxwvz"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.bot.classify(text)
        self.assertEqual(self.bot.summarize("  "), "")
        self.assertEqual(self.bot.search("qzxwvz"), [])
        self.assertEqual(detect_language(""), ("unknown", 0.0))
        self.assertEqual(translate("", "es", "en"), ("", False))

    def test_streamlit_pages(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=80).run()
        self.assertFalse(app.exception or app.error)
        app.button[0].click().run()
        self.assertTrue(app.warning)
        app.text_area[0].set_value(ARTICLE)
        app.button[0].click().run()
        self.assertFalse(app.exception or app.error)
        self.assertEqual(app.metric[0].value, self.bot.classify(ARTICLE)[0][0].title())
        team = next(item.value for item in app.sidebar.markdown if "Marisabel Morales" in item.value)
        self.assertEqual(team.split(), ["Marisabel", "Morales", "Khalida", "Bestani", "Sadea", "Saeed"])
        app.radio[0].set_value("Summarize & Search").run()
        app.text_area[0].set_value(ARTICLE)
        app.button[0].click().run()
        self.assertTrue(any(h.value == "TextRank summary" for h in app.subheader))
        app.text_input[0].set_value("elections and voters")
        app.button[1].click().run()
        self.assertTrue(any("similarity" in e.label for e in app.expander))
        if self.bot.is_demo:
            self.assertTrue(any("Original and contributor history" in item.proto.label
                                for item in app.get("link_button")))
        app.radio[0].set_value("Multilingual Analysis").run()
        app.text_area[0].set_value(ENGLISH)
        app.button[0].click().run()
        self.assertFalse(app.exception or app.error)
        self.assertIn("English", str(app.dataframe[0].value))
        app.radio[0].set_value("Chat with NewsBot").run()
        app.chat_input[0].set_value("Show me positive tech articles").run()
        app.chat_input[0].set_value("More").run()
        self.assertTrue(any("Next results" in t.value for t in app.text))
        if self.bot.is_demo:
            self.assertTrue(any(e.label == "Wikinews sources and contributors" for e in app.expander))
        self.assertFalse(app.exception or app.error)
