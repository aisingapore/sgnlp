import unittest
from unittest.mock import MagicMock, call

from sgnlp.base.preprocessor import PreprocessorBase


class DummyPreprocessor(PreprocessorBase):
    """This is a dummy preprocessor class for testing.
    It takes in data of questions and answers pairs,
    and does a simple white space splitting for tokenizing,
    and converts them to a field, the length of each token, respectively.
    """

    def preprocess(self, data):
        tokenized_questions = [q.split(" ") for q in data["questions"]]
        tokenized_answers = [a.split(" ") for a in data["answers"]]

        questions_lengths = []
        for question in tokenized_questions:
            question_lengths = [len(token) for token in question]
            questions_lengths.append(question_lengths)

        answers_lengths = []
        for answer in tokenized_answers:
            answer_lengths = [len(token) for token in answer]
            answers_lengths.append(answer_lengths)

        processed_data = {
            "questions_lengths": questions_lengths,
            "answers_lengths": answers_lengths
        }

        return processed_data


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.data = {
            "questions": [
                "What color is the sky",
                "What is the weather today",
                "When will winter come"
            ],
            "answers": [
                "Blue",
                "I don't know",
                "Never"
            ],
        }

        self.expected_processed_data = {
            "questions_lengths": [
                [4, 5, 2, 3, 3],
                [4, 2, 3, 7, 5],
                [4, 4, 6, 4]
            ],
            "answers_lengths": [
                [4],
                [1, 5, 4],
                [5]
            ]
        }

    def test_preprocessor_processes_correctly(self):
        preprocessor = DummyPreprocessor()
        processed_data = preprocessor(self.data)

        self.assertEqual(processed_data, self.expected_processed_data)

    def test_preprocessor_errors_out_when_input_data_is_not_of_equal_length(self):
        test_data = {
            "questions": [
                "What color is the sky"
            ],
            "answers": ["Blue", "I don't know"]
        }
        preprocessor = DummyPreprocessor()
        with self.assertRaises(ValueError) as context:
            preprocessor(test_data)

    def test_preprocessor_processes_in_batches(self):
        preprocessor = DummyPreprocessor(batch_size=2)

        # Makes function a mock but retains actual implementation
        preprocessor.preprocess = MagicMock(side_effect=preprocessor.preprocess)

        preprocessor(self.data)

        # Given data of size 3 and batch size of 2, expect to process 2 data points followed by 1 data point
        preprocessor.preprocess.assert_has_calls(
            calls=[call({'questions': ['What color is the sky', 'What is the weather today'],
                         'answers': ['Blue', "I don't know"]}),
                   call({'questions': ['When will winter come'], 'answers': ['Never']})],
            any_order=False
        )
