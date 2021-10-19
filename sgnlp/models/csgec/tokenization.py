from transformers import GPT2Tokenizer
import re


def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class CsgTokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize(self, text):
        bpe_tokens = []
        for token in text.split(" "):
            token = token.lstrip(" ")
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            new_tokens = [bpe_token + "@@" for bpe_token in self.bpe(token).split(" ")]
            new_tokens[-1] = new_tokens[-1].rstrip("@@")
            bpe_tokens.extend(new_tokens)
        return bpe_tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        assert (
            token_ids_1 is None
        ), "This model does not support paired sentences. Please check your inputs"

        return token_ids_0 + [self.eos_token_id]

    def bpe(self, token):

        if token in self.cache:
            return self.cache[token]

        word = tuple(token) + ("</w>",)
        pairs = get_pairs(word)

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # don't print end-of-word symbols
        if word[-1] == "</w>":
            word = word[:-1]
        elif word[-1].endswith("</w>"):
            word = word[:-1] + (word[-1].replace("</w>", ""),)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = ""
        for token in tokens[:-1]:
            text += bytearray([self.byte_decoder[c] for c in token]).decode(
                "utf-8", errors=self.errors
            )
            text += " "
        text += bytearray([self.byte_decoder[c] for c in tokens[-1]]).decode(
            "utf-8", errors=self.errors
        )
        text = (
            text.replace(",@@ ", ", ")
            .replace("@@ ", "")
            .replace(" n't", "n't")
            .replace(" 'd ", "'d ")
            .replace(" 's ", "'s ")
            .replace(" 'm ", "'m ")
            .replace(" 'll ", "'ll ")
            .replace(" 're ", "'re ")
            .replace(" 've ", "'ve ")
        )
        text = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", text)
        return text
