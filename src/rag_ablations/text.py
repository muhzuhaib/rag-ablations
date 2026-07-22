"""Tokenisation.

The analyzer is a benchmark variable, not a fixed detail. Published BM25
baselines (BEIR's are Anserini/Lucene) stem tokens and drop stopwords, so a
naive whitespace tokeniser scores several points lower and makes our BM25 look
worse than it is. Rather than hide that, both analyzers are available and the
results table reports each.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[a-z0-9]+")

# Lucene's English stopword list, which is what the published baselines use.
STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with""".split()
)

_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, optionally stem and filter."""
    tokens = _WORD.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens
