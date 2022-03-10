import unittest

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
