import re


def prepare_output_sentence(sent):
    sent = sent.replace(",@@ ", ", ")
    sent = sent.replace("@@ ", "")
    sent = sent.replace(" n't", "n't")
    sent = sent.replace(" 'd ", "'d ")
    sent = sent.replace(" 's ", "'s ")
    sent = sent.replace(" 'm ", "'m ")
    sent = sent.replace(" 'll ", "'ll ")
    sent = sent.replace(" 're ", "'re ")
    sent = sent.replace(" 've ", "'ve ")
    sent = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", sent)
    return sent


class CsgecPostprocessor:
    def __init__(self, tgt_tokenizer):
        self.tgt_tokenizer = tgt_tokenizer

    def __call__(self, batch_predicted_ids):
        batch_predicted_sentences = []
        for text_predicted_ids in batch_predicted_ids:
            text_predicted_sentences = []
            for sentence_predicted_ids in text_predicted_ids:
                text_predicted_sentences.append(
                    prepare_output_sentence(
                        self.tgt_tokenizer.decode(sentence_predicted_ids)
                    )
                )
            batch_predicted_sentences.append(" ".join(text_predicted_sentences))

        return batch_predicted_sentences
