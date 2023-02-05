import unittest

import load_data
from LDA import LDA
from QDA import QDA
from RegressionLineaire import RegressionLineaire
from RegressionLogistique import RegressionLogistique


class TestLoadDataMethods(unittest.TestCase):
    def test_main(self):
        self.assertTrue(load_data.main())


class TestRegressionLineaireMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = RegressionLineaire()
        self.assertTrue(my_classifier.main())


class TestRegressionLogistiqueMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = RegressionLogistique()
        self.assertTrue(my_classifier.main())


class TestLDAMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = LDA()
        self.assertTrue(my_classifier.main())


class TestQDAMethods(unittest.TestCase):
    def test_main(self):
        my_classifier = QDA()
        self.assertTrue(my_classifier.main())


if __name__ == '__main__':
    unittest.main()
