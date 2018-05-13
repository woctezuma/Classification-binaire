import unittest

import load_data
import main
from LDA import LDA
from QDA import QDA
from RegressionLineaire import RegressionLineaire
from RegressionLogistique import RegressionLogistique


class TestLoadDataMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(load_data.main())


class TestRegressionLineaireMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(RegressionLineaire.main())


class TestRegressionLogistiqueMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(RegressionLogistique.main())


class TestLDAMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(LDA.main())


class TestQDAMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(QDA.main())


class TestMainMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(main.main())


if __name__ == '__main__':
    unittest.main()
