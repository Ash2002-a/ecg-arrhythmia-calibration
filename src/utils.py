# src/utils.py
from __future__ import annotations

# AAMI EC57-ish 5-class mapping used widely in MIT-BIH papers.
# Anything not in the map can be dropped or mapped to Q depending on your plan.

AAMI_MAP = {
    # N class (Normal)
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",

    # S class (Supraventricular ectopic)
    "A": "S", "a": "S", "J": "S", "S": "S",

    # V class (Ventricular ectopic)
    "V": "V", "E": "V",

    # F class (Fusion)
    "F": "F",

    # Q class (Unknown / paced / other)
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
    "P": "Q", "p": "Q",  # paced beats
    "|": "Q",
}

# Non-beat annotation markers you should ignore (not actual beats)
NON_BEAT_SYMBOLS = {"+", "~", "[", "]", "!", "x", "(", ")", "`", "^", "{", "}", "t", "u"}