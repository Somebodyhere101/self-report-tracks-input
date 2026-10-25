"""Automatic behaviour scorers for S2 (frozen with the preregistration).

german_score(text) -> (is_german: bool, g_ratio: float, n_function_words: int)
  Counts German and English function words (words that are common in both
  languages, such as in/so/man/an/was/will/also/war/hat/her, are in neither list).
  German iff at least 3 function words were found and >= 50% of them are German.
caps_score(text) -> (is_caps: bool, upper_fraction: float, n_letters: int)
  All-caps iff at least 20 letters and >= 80% of letters are upper case.
"""
import re

GERMAN_FW = set("""der die das und ist nicht ein eine einen einem einer eines es ich sie wir ihr du zu mit
von auf für sich auch als im den dem des sind wird werden kann können haben wie oder aber wenn bei nach
aus noch nur über um durch dass mehr sehr gibt sein seine diese dieser dieses viele oft zum zur vom beim
ihre ihren doch schon weil damit kein keine wo wer hier dort unter zwischen gegen ohne sowie bereits etwa
immer jedoch sowohl jede jeder jedes ihn ihm uns euch wurde wurden ob""".split())
ENGLISH_FW = set("""the and is are of to a that it for on with as be by this not or but you we they he she
can have from at which more very there their these those its your our been were would could should about
into many often some other such when what how if than them then because while only just all one do does
who where why""".split())
assert not (GERMAN_FW & ENGLISH_FW)
WORD = re.compile(r"[a-zäöüß]+")


def german_score(text):
    words = WORD.findall(text.lower())
    ng = sum(w in GERMAN_FW for w in words)
    ne = sum(w in ENGLISH_FW for w in words)
    n = ng + ne
    g = ng / n if n else 0.0
    return (n >= 3 and g >= 0.5), g, n


def caps_score(text):
    letters = [c for c in text if c.isalpha()]
    n = len(letters)
    up = sum(c.isupper() for c in letters) / n if n else 0.0
    return (n >= 20 and up >= 0.8), up, n


SCORERS = {"german": german_score, "caps": caps_score}
