import math
from typing import List


class StatisticsEngine:

    @staticmethod
    def mean(values: List[float]):

        if not values:
            return 0

        return sum(values) / len(values)

    @staticmethod
    def std(values: List[float]):

        if len(values) <= 1:
            return 0

        m = StatisticsEngine.mean(values)

        variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)

        return math.sqrt(variance)

    @staticmethod
    def confidence_interval(values: List[float], z: float = 1.96):

        if not values:
            return (0, 0)

        m = StatisticsEngine.mean(values)
        s = StatisticsEngine.std(values)

        margin = z * (s / math.sqrt(len(values)))

        return (m - margin, m + margin)