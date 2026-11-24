"""Notebook language detection and Google → MyMemory translation fallback."""

import logging
import re

import requests
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0
LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ar": "Arabic", "zh-cn": "Chinese",
}
LOCALES = {
    "en": "en-GB", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
    "it": "it-IT", "pt": "pt-PT", "ar": "ar-SA", "zh-cn": "zh-CN",
}
log = logging.getLogger(__name__)


def detect_language(text):
    try:
        best = detect_langs(text.strip())[0]
        return best.lang, round(best.prob, 3)
    except LangDetectException:
        return "unknown", 0.0


def chunks(text, byte_limit):
    """Split at whitespace where possible and respect UTF-8 service limits."""
    while len(text.encode("utf-8")) > byte_limit:
        prefix = text.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")
        match = list(re.finditer(r"\s+", prefix))
        cut = match[-1].start() if match and match[-1].start() else len(prefix)
        yield text[:cut]
        text = text[cut:].lstrip()
    if text:
        yield text


def google(session, text, source, target):
    response = session.get("https://translate.google.com/m", params={
        "sl": source, "tl": target, "q": text,
    }, timeout=(3, 5))
    response.raise_for_status()
    result = BeautifulSoup(response.text, "html.parser").select_one(".result-container, .t0")
    return result.get_text(strip=True) if result else ""


def mymemory(session, text, source, target):
    response = session.get("https://api.mymemory.translated.net/get", params={
        "q": text, "langpair": f"{LOCALES[source]}|{LOCALES[target]}",
    }, timeout=(3, 5))
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("responseStatus")) == "200":
        return payload.get("responseData", {}).get("translatedText", "")
    return ""


def translate(text, target="en", source="auto"):
    """Return complete translated text, or an explicit failure with no result."""
    text = text.strip()
    source, target = source.lower(), target.lower()
    if not text or target not in LANG_NAMES or len(text) > 4500:
        return "", False
    if source == "auto":
        language, _ = detect_language(text)
        source = language if language != "unknown" else "auto"
    if source == target:
        return text, True

    providers = [("Google", google, 4500)]
    if source in LOCALES:
        providers.append(("MyMemory", mymemory, 480))
    with requests.Session() as session:
        for name, provider, limit in providers:
            try:
                output = []
                for chunk in chunks(text, limit):
                    translated = provider(session, chunk, source, target).strip()
                    if not translated:
                        break
                    output.append(translated)
                else:
                    result = " ".join(output)
                    if " ".join(result.split()).casefold() != " ".join(text.split()).casefold():
                        return result, True
            except (requests.RequestException, ValueError, TypeError, AttributeError) as error:
                log.info("Translation provider %s failed (%s)", name, type(error).__name__)
    return "", False
