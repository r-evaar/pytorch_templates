import numpy as np


class haversine:

    def __init__(self, r):
        """
        Creates a haversine formula for a sphere

        functor = haversine(r)

        :param r: The radius of the sphere
        :return: A functor of a formula to calculate the haversine angular distance for a sphere of radius r
        """
        self.r = r

    def __call__(self, point1, point2):
        f"""
        Calculates the haversine angular distance between two points, 
        in the geographic coordinate system, on the surface of a sphere 
        with radius{self.r}
        :param point1: First point in degrees (longitude1, latitude1)
        :param point2: Second point in degrees (longitude2, latitude2)
        :return: angular distance - numeric value
        """

        phi1 = np.radians(point1[1])
        phi2 = np.radians(point2[1])

        d_phi = np.radians(point2[1] - point1[1])
        d_lambda = np.radians(point2[0]-point1[0])

        under_sqrt = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2

        return 2 * self.r * np.arcsin(np.sqrt(under_sqrt))


