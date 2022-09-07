import unittest
from utils.measurements import *


class TestMeasurements(unittest.TestCase):

    def test_haversine(self):
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug
        haversine_on_earth = haversine(6.3781e3)  # Radius in km
        points_1 = (np.asarray([-73.992365, -73.990078, -73.994149]), np.asarray([40.730521, 40.740558, 40.751118]))
        points_2 = (np.asarray([-73.975499, -73.974232, -73.960064]), np.asarray([40.744746, 40.744114, 40.766235]))
        result = haversine_on_earth(points_1, points_2).round(2).tolist()
        expected = [2.13, 1.39, 3.33]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
