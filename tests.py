import unittest

import load_data
from LDA import LDA
from QDA import QDA
from RegressionLineaire import RegressionLineaire
from RegressionLogistique import RegressionLogistique


class TestLoadDataMethods(unittest.TestCase):
    def test_main(self):
        assert load_data.main() is True


class TestRegressionLineaireMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = RegressionLineaire()
        assert my_classifier.main() is True


class TestRegressionLogistiqueMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = RegressionLogistique()
        assert my_classifier.main() is True


class TestLDAMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = LDA()
        assert my_classifier.main() is True


class TestQDAMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = QDA()
        assert my_classifier.main() is True


if __name__ == '__main__':
    unittest.main()
