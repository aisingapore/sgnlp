import unittest

import torch

from sgnlp.models.sentic_gcn.modeling import SenticGCNModelOutput, SenticGCNBertModelOutput
from sgnlp.models.sentic_gcn.preprocess import SenticGCNData, SenticGCNBertData
from sgnlp.models.sentic_gcn.postprocess import SenticGCNPostprocessor, SenticGCNBertPostprocessor


class TestSenticGCNPostprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_processed_inputs = [
            SenticGCNData(
                full_text="soup is tasty but soup is a little salty. salty funkysoup.",
                aspect="soup",
                left_text="",
                full_text_tokens=[
                    "Soup",
                    "is",
                    "tasty",
                    "but",
                    "soup",
                    "is",
                    "a",
                    "little",
                    "salty.",
                    "Salty",
                    "funkysoup.",
                ],
                aspect_token_indexes=[0],
            ),
            SenticGCNData(
                full_text="soup is tasty but soup is a little salty. salty funkysoup.",
                aspect="soup",
                left_text="soup is tasty but",
                full_text_tokens=[
                    "Soup",
                    "is",
                    "tasty",
                    "but",
                    "soup",
                    "is",
                    "a",
                    "little",
                    "salty.",
                    "Salty",
                    "funkysoup.",
                ],
                aspect_token_indexes=[4],
            ),
            SenticGCNData(
                full_text="everyone that sat in the back outside agreed that it was the worst service we had ever received.",
                aspect="service",
                left_text="everyone that sat in the back outside agreed that it was the worst",
                full_text_tokens=[
                    "Everyone",
                    "that",
                    "sat",
                    "in",
                    "the",
                    "back",
                    "outside",
                    "agreed",
                    "that",
                    "it",
                    "was",
                    "the",
                    "worst",
                    "service",
                    "we",
                    "had",
                    "ever",
                    "received.",
                ],
                aspect_token_indexes=[13],
            ),
            SenticGCNData(
                full_text="it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more .",
                aspect="location",
                left_text="it 's located in a strip mall near the beverly center , not the greatest",
                full_text_tokens=[
                    "it",
                    "'s",
                    "located",
                    "in",
                    "a",
                    "strip",
                    "mall",
                    "near",
                    "the",
                    "beverly",
                    "center",
                    ",",
                    "not",
                    "the",
                    "greatest",
                    "location",
                    ",",
                    "but",
                    "the",
                    "food",
                    "keeps",
                    "me",
                    "coming",
                    "back",
                    "for",
                    "more",
                    ".",
                ],
                aspect_token_indexes=[15],
            ),
            SenticGCNData(
                full_text="it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more .",
                aspect="food",
                left_text="it 's located in a strip mall near the beverly center , not the greatest location , but the",
                full_text_tokens=[
                    "it",
                    "'s",
                    "located",
                    "in",
                    "a",
                    "strip",
                    "mall",
                    "near",
                    "the",
                    "beverly",
                    "center",
                    ",",
                    "not",
                    "the",
                    "greatest",
                    "location",
                    ",",
                    "but",
                    "the",
                    "food",
                    "keeps",
                    "me",
                    "coming",
                    "back",
                    "for",
                    "more",
                    ".",
                ],
                aspect_token_indexes=[19],
            ),
        ]
        self.test_model_outputs = SenticGCNModelOutput(
            loss=None,
            logits=torch.ones([5, 3], dtype=torch.float32),
        )

    def test_senticgcn_postprocess(self):
        post_proc = SenticGCNPostprocessor()
        post_outputs = post_proc(processed_inputs=self.test_processed_inputs, model_outputs=self.test_model_outputs)
        self.assertEqual(len(post_outputs), 3)
        for key in ["sentence", "aspects", "labels"]:
            for output in post_outputs:
                self.assertTrue(key in output.keys())
        self.assertEqual(len(post_outputs[0]["aspects"]), 2)
        self.assertEqual(len(post_outputs[1]["aspects"]), 1)
        self.assertEqual(len(post_outputs[2]["aspects"]), 2)
        self.assertEqual(len(post_outputs[0]["labels"]), 2)
        self.assertEqual(len(post_outputs[1]["labels"]), 1)
        self.assertEqual(len(post_outputs[2]["labels"]), 2)

    def test_senticgcn_post_process_return_text_and_aspect(self):
        post_proc = SenticGCNPostprocessor(return_full_text=True, return_aspects_text=True)
        post_outputs = post_proc(processed_inputs=self.test_processed_inputs, model_outputs=self.test_model_outputs)
        for key in ["sentence", "aspects", "labels", "full_text", "aspects_text"]:
            for output in post_outputs:
                self.assertTrue(key in output.keys())


class TestSenticGCNBertPostprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_processed_inputs = [
            SenticGCNBertData(
                full_text="soup is tasty but soup is a little salty. salty funkysoup.",
                aspect="soup",
                left_text="",
                full_text_with_bert_tokens="[CLS] soup is tasty but soup is a little salty. salty funkysoup. [SEP] soup [SEP]",
                full_text_tokens=[
                    "Soup",
                    "is",
                    "tasty",
                    "but",
                    "soup",
                    "is",
                    "a",
                    "little",
                    "salty.",
                    "Salty",
                    "funkysoup.",
                ],
                aspect_token_indexes=[0],
            ),
            SenticGCNBertData(
                full_text="soup is tasty but soup is a little salty. salty funkysoup.",
                aspect="soup",
                left_text="soup is tasty but",
                full_text_with_bert_tokens="[CLS] soup is tasty but soup is a little salty. salty funkysoup. [SEP] soup [SEP]",
                full_text_tokens=[
                    "Soup",
                    "is",
                    "tasty",
                    "but",
                    "soup",
                    "is",
                    "a",
                    "little",
                    "salty.",
                    "Salty",
                    "funkysoup.",
                ],
                aspect_token_indexes=[4],
            ),
            SenticGCNBertData(
                full_text="everyone that sat in the back outside agreed that it was the worst service we had ever received.",
                aspect="service",
                left_text="everyone that sat in the back outside agreed that it was the worst",
                full_text_with_bert_tokens="[CLS] everyone that sat in the back outside agreed that it was the worst service we had ever received. [SEP] service [SEP]",
                full_text_tokens=[
                    "Everyone",
                    "that",
                    "sat",
                    "in",
                    "the",
                    "back",
                    "outside",
                    "agreed",
                    "that",
                    "it",
                    "was",
                    "the",
                    "worst",
                    "service",
                    "we",
                    "had",
                    "ever",
                    "received.",
                ],
                aspect_token_indexes=[13],
            ),
            SenticGCNBertData(
                full_text="it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more .",
                aspect="location",
                left_text="it 's located in a strip mall near the beverly center , not the greatest",
                full_text_with_bert_tokens="[CLS] it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more . [SEP] location [SEP]",
                full_text_tokens=[
                    "it",
                    "'s",
                    "located",
                    "in",
                    "a",
                    "strip",
                    "mall",
                    "near",
                    "the",
                    "beverly",
                    "center",
                    ",",
                    "not",
                    "the",
                    "greatest",
                    "location",
                    ",",
                    "but",
                    "the",
                    "food",
                    "keeps",
                    "me",
                    "coming",
                    "back",
                    "for",
                    "more",
                    ".",
                ],
                aspect_token_indexes=[15],
            ),
            SenticGCNBertData(
                full_text="it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more .",
                aspect="food",
                left_text="it 's located in a strip mall near the beverly center , not the greatest location , but the",
                full_text_with_bert_tokens="[CLS] it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more . [SEP] food [SEP]",
                full_text_tokens=[
                    "it",
                    "'s",
                    "located",
                    "in",
                    "a",
                    "strip",
                    "mall",
                    "near",
                    "the",
                    "beverly",
                    "center",
                    ",",
                    "not",
                    "the",
                    "greatest",
                    "location",
                    ",",
                    "but",
                    "the",
                    "food",
                    "keeps",
                    "me",
                    "coming",
                    "back",
                    "for",
                    "more",
                    ".",
                ],
                aspect_token_indexes=[19],
            ),
        ]
        self.test_model_outputs = SenticGCNBertModelOutput(
            loss=None,
            logits=torch.ones([5, 3], dtype=torch.float32),
        )

    def test_senticgcnbert_postprocess(self):
        post_proc = SenticGCNBertPostprocessor()
        post_outputs = post_proc(processed_inputs=self.test_processed_inputs, model_outputs=self.test_model_outputs)
        self.assertEqual(len(post_outputs), 3)
        for key in ["sentence", "aspects", "labels"]:
            for output in post_outputs:
                self.assertTrue(key in output.keys())
        self.assertEqual(len(post_outputs[0]["aspects"]), 2)
        self.assertEqual(len(post_outputs[1]["aspects"]), 1)
        self.assertEqual(len(post_outputs[2]["aspects"]), 2)
        self.assertEqual(len(post_outputs[0]["labels"]), 2)
        self.assertEqual(len(post_outputs[1]["labels"]), 1)
        self.assertEqual(len(post_outputs[2]["labels"]), 2)

    def test_senticgcn_post_process_return_text_and_aspect(self):
        post_proc = SenticGCNBertPostprocessor(return_full_text=True, return_aspects_text=True)
        post_outputs = post_proc(processed_inputs=self.test_processed_inputs, model_outputs=self.test_model_outputs)
        for key in ["sentence", "aspects", "labels", "full_text", "aspects_text"]:
            for output in post_outputs:
                self.assertTrue(key in output.keys())
