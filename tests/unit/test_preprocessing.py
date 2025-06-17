import unittest
from pathlib import Path
from app.services import extract_top_n_nouns_with_frequency
from app.services.preprocessing import _basic_clean_text

import tempfile
import fitz  # PyMuPDF
import os
import nltk

from app.utils.pdf_file_utils import pdf2text

# Ensure required NLTK resources are downloaded
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

class TestTextPreprocessor(unittest.TestCase):

    def test_basic_clean_text(self):
        raw_text = "Hello, World! This is a test. 123"
        expected = "hello world this is a test 123"
        cleaned = _basic_clean_text(raw_text)
        self.assertEqual(cleaned, expected)

    def test_extract_top_n_nouns_with_frequency(self):
        text = "The cat sat on the mat. The dog barked at the cat. The gardener planted plants in the garden."
        result = extract_top_n_nouns_with_frequency(text, top_n=5)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("cat", result)
        self.assertEqual(result["cat"], 2)

    def test_extract_top_n_nouns_invalid_input(self):
        with self.assertRaises(ValueError):
            extract_top_n_nouns_with_frequency("", top_n=10)

        with self.assertRaises(ValueError):
            extract_top_n_nouns_with_frequency(12345, top_n=10)

    def test_pdf2text_valid_pdf(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_pdf_path = Path(tmpdirname) / "test.pdf"

            # Create a simple 1-page PDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "This is a test PDF with some text.")
            doc.save(temp_pdf_path)

            extracted_text = pdf2text(temp_pdf_path)
            self.assertIn("test PDF", extracted_text)

    def test_pdf2text_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            pdf2text(Path("nonexistent_file.pdf"))

    def test_pdf2text_invalid_file_type(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
            tmp.write("Just a plain text file.")
            tmp_path = Path(tmp.name)

        with self.assertRaises(ValueError):
            pdf2text(tmp_path)

        os.remove(tmp_path)

class TestTextPreprocessorComplexCase(unittest.TestCase):
    def test_extract_top_n_nouns_complex_text(self):
        sample_text = """
            In the **conference room**, the CEO addressed the company. The company values—integrity, innovation,
            and impact—were emphasized throughout the presentation. Employees from different departments like 
            marketing, engineering, and human resources joined the discussion. The presentation included data, 
            charts, and projections. The CEO mentioned "data" and "innovation" multiple times, while the team 
            leads contributed with insights about performance and strategy. BUG! BUG! BUG! BUG! BUG!
        """

        result = extract_top_n_nouns_with_frequency(sample_text, top_n=5)

        # Assertions: focus on correct nouns and order by frequency
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(len(result), 3)

        expected_nouns = {"bug", "company", "innovation", "data", "presentation", "ceo"}
        found_nouns = set(result.keys())

        # Check that expected frequent nouns are in the result
        self.assertTrue(expected_nouns.intersection(found_nouns))

        # Check that only nouns are returned (heuristic, not bulletproof)
        for noun in result:
            self.assertTrue(noun.isalpha())

if __name__ == "__main__":
    unittest.main()
