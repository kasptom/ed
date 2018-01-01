from unittest import TestCase
from src.preprocessing.configuration import print_configuration


class TestPrintConfiguration(TestCase):

    def test_print_configuration(self):
        print_configuration()
