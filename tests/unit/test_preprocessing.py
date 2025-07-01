import unittest
from src.services.preprocessing import (
    _basic_clean_text, 
    extract_top_n_nouns_with_frequency
)

class TestPreprocessing(unittest.TestCase):

    def test_basic_clean_text(self):
        text = "Hey Fellas!! How are you? Hey, David! Are you going to the market?!!"
        expected = "hey fellas how are you hey david are you going to the market"
        self.assertEqual(_basic_clean_text(text), expected,
                         "Expected text and cleaned text should be equal!")

    def test_extract_top_n_nouns_with_frequency(self):
        text = "The cat sat on the mat. The dog chased the cat. The cat ran."
        stop_words = {"the", "on"}
        all_nouns = {"cat", "dog", "mat"}
        top_n = 2

        result = extract_top_n_nouns_with_frequency(text, top_n, stop_words, all_nouns)
        print("Result:", result)

        # Must contain 'cat': 3, and either 'mat': 1 or 'dog': 1
        self.assertEqual(result.get('cat'), 3)
        self.assertTrue(
            ('mat' in result and result['mat'] == 1) or
            ('dog' in result and result['dog'] == 1),
            f"Expected second noun to be 'mat' or 'dog', got: {result}"
        )

    def test_extract_top_n_nouns_with_frequency_invalid_input(self):
        with self.assertRaises(ValueError):
            extract_top_n_nouns_with_frequency("", 3, set(), set())

        with self.assertRaises(ValueError):
            extract_top_n_nouns_with_frequency(None, 3, set(), set())

    def test_extract_top_n_nouns_with_frequency_complex(self):
        text = (
            "In the laboratory, the scientist observed a reaction. "
            "The reaction involved chemicals and compounds. "
            "The scientist, wearing a white lab coat, noted changes in the solution. "
            "The lab assistant recorded the results in the notebook. "
            "Both the scientist and assistant discussed the findings."
        )

        stop_words = {"the", "in", "a", "and", "of"}
        all_nouns = {
            "laboratory", "scientist", "reaction", "chemicals", "compounds",
            "lab", "coat", "solution", "assistant", "results", "notebook", "findings"
        }

        top_n = 5

        result = extract_top_n_nouns_with_frequency(text, top_n, stop_words, all_nouns)
        print("Complex Test Result:", result)

        # Must have the correct top frequent nouns
        self.assertEqual(result.get("scientist"), 3)
        self.assertEqual(result.get("reaction"), 2)
        self.assertEqual(result.get("lab"), 2)
        self.assertEqual(result.get("assistant"), 2)

        # Fifth noun can be either 'laboratory' or 'results'
        fifth_noun = set(result.keys()) - {"scientist", "reaction", "lab", "assistant"}
        self.assertTrue(
            fifth_noun.issubset({"laboratory", "results"}),
            f"Unexpected fifth noun: {fifth_noun}"
        )


if __name__ == '__main__':
    unittest.main()
