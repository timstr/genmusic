import numpy as np
import math


class LossPlotter:
    def __init__(self, aggregation_interval, colour, num_quantiles):
        assert isinstance(aggregation_interval, int)
        r, g, b = colour
        assert isinstance(r, float)
        assert isinstance(g, float)
        assert isinstance(b, float)
        assert isinstance(num_quantiles, int)
        assert num_quantiles >= 1 and num_quantiles < aggregation_interval
        self._colour_r = r
        self._colour_g = g
        self._colour_b = b
        self._aggregation_interval = aggregation_interval
        self._colour = colour
        self._num_quantiles = num_quantiles
        self._values = [[] for _ in range(num_quantiles + 1)]
        self._acc = []

    def append(self, item):
        assert isinstance(item, float)
        if math.isnan(item) or math.isnan(item):
            item = 0.0
        self._acc.append(item)
        if len(self._acc) == self._aggregation_interval:
            q = np.linspace(0.0, 1.0, num=(self._num_quantiles + 1))
            qv = np.quantile(self._acc, q)
            for i in range(self._num_quantiles + 1):
                self._values[i].append(qv[i])
            self._acc = []

    def plot_to(self, plt_axis):
        colour_dark = (self._colour_r, self._colour_g, self._colour_b)
        if self._aggregation_interval > 1:
            x_min = self._aggregation_interval // 2
            x_stride = self._aggregation_interval
            x_count = len(self._values[0])
            x_values = range(x_min, x_min + x_count * x_stride, x_stride)
            for i in range(self._num_quantiles):
                t = i / self._num_quantiles
                t = 2.0 * min(t, 1.0 - t)
                c = (self._colour_r, self._colour_g, self._colour_b, t)
                plt_axis.fill_between(
                    x=x_values, y1=self._values[i], y2=self._values[i + 1], color=c
                )
        else:
            plt_axis.scatter(
                range(self._values[0]), self._items, s=1.0, color=colour_dark
            )

    def save(self, path):
        data = np.array(self._values)
        with open(path, "wb") as f:
            np.save(f, data)

    def load(self, path):
        with open(path, "rb") as f:
            data = np.load(f)
        q, l = data.shape
        assert q == self._num_quantiles + 1
        self._values = [list(a) for a in data]
        self._acc = []
