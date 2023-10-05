import panel as pn
import numpy as np
from panel.viewable import Viewer


class Component(Viewer):
    # watch these for changes
    # data_series = param.Parameter()

    def __init__(self, data_series, title, **params):
        self.data = {'x': np.arange(len(data_series)),
                     'y': data_series.tolist()}
        self.title = title
        self.trend = pn.indicators.Trend(
            name=title, data=self.data, sizing_mode='stretch_both')
        super().__init__(**params)

    def stream_data(self, entries: list):
        self.trend.stream(
            {'x': [self.trend.data['x'][-1]+1], 'y': entries}, rollover=60)

    def __panel__(self):
        self._layout = pn.Column(
            pn.pane.Str(self.title),
            self.trend, sizing_mode='stretch_both')
        return self._layout