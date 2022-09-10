import unittest
import pandas as pd
from src.preprocess import pd_to_torch
import torch


class TestPreProcess(unittest.TestCase):

    def test_pd_to_torch(self):
        print("Loading test dataset .. ", end="")
        test_set = pd.read_csv('../data/NYCTaxiFares.csv')
        print("done")

        cat_f = ['passenger_count']
        cont_f = ['pickup_longitude', 'pickup_latitude']
        y_f = 'fare_amount'
        results = pd_to_torch(test_set, cat_f, cont_f, y_f)

        self.assertIs(type(results[0]), tuple, msg="Categorical output is not a (values, e_sizes) tuple")
        self.assertTrue(torch.is_tensor(results[0][0]), msg="Categorical values are not a tensor")
        self.assertTrue(torch.is_tensor(results[1]), msg="Continuous values are not a tensor")
        self.assertTrue(torch.is_tensor(results[2]), msg="Output values are not a tensor")


if __name__ == '__main__':
    unittest.main()
