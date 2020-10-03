
import sentencepiece as spm

import unicodedata as ud


# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/google-research/bert
# which is under the Apache License 2.0



class WordPieceParser(object):

    def __init__(self, wp_model_path):

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(wp_model_path)

        self.vocab_size = self._sp.get_piece_size()


    def parse(self, text, parse_format='bag-of-words'):

        wp_ids = self._sp.encode_as_ids(text_normalize(text))

        if parse_format=='sequence':
            return wp_ids

        else:

            bagofwords = {}

            for wp_id in wp_ids:
                bagofwords.setdefault(wp_id, 0)
                bagofwords[wp_id] += 1

            return bagofwords


def text_normalize(text, to_lower=True, unorm='NFKD'):

    # for sentecepiece的前處理，不特別對CJK和標點進行處理

    output = []

    text = text.lower() if to_lower else text

    for char in ud.normalize(unorm, text):

        cp = ord(char)

        if (cp == 0 or cp == 0xfffd or _is_control(char) or 
           ud.category(char)=='Mn'):
            continue

        elif _is_whitespace(char):
            output.append(" ")

        else:
            output.append(char)

    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    return char in [' ', '\t', '\n', '\r'] or ud.category(char)=='Zs'

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    return char not in ['\t', '\n', '\r'] and ud.category(char) in ('Cc', 'Cf')
