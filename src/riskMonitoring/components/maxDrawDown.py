
import panel as pn
import plotly.express as px
from panel.viewable import Viewer
import param
import riskMonitoring.styling as styling
import plotly.graph_objs as go
import riskMonitoring.utils as utils
import riskMonitoring.processing as processing


class Component(Viewer):
    selected_key_column = param.Parameter()
    calcualted_p_stock = param.Parameter()
    start_date = param.Parameter()
    end_date = param.Parameter()

    def __init__(self, styles, calculated_p_stock, max_width, min_width, **params):
        self.styles = styles
        self.max_width = max_width
        self.min_width = min_width
        self.date_range = pn.widgets.DateRangeSlider(
            start=calculated_p_stock.time.min(),
            end=calculated_p_stock.time.max(),
            value=(calculated_p_stock.time.min(),
                   calculated_p_stock.time.max())
        )

        self.calculated_p_stock = calculated_p_stock
        self._sycn_params()
        self.return_dd_plot = pn.pane.Plotly(
            self.plot_drawdown('cum_return_dd'))
        super().__init__(**params)

    @param.depends('date_range.value', watch=True)
    def _sycn_params(self):
        self.start_date = self.date_range.value[0]
        self.end_date = self.date_range.value[1]

    def plot_drawdown(self, on='cum_return_dd'):
        '''
        plot either cum_return_dd or cum_pnl_dd
        '''
        cliped_df = self.calculated_p_stock[self.calculated_p_stock.time.between(
            self.start_date, self.end_date, inclusive='both')]

        df = processing.get_draw_down(cliped_df)
        df.period = df.period.dt.strftime('%Y-%m-%d')
        fig = px.line(df, x='period', y=[on])

        # add scatter to represetn new high
        ex_max = 'ex_max_cum_return' if on == 'cum_return_dd' else 'ex_max_cum_pnl'
        cur = 'cum_return' if on == 'cum_return_dd' else 'cum_pnl'
        new_height_pnl = df[df[ex_max] == df[cur]]
        fig.add_trace(go.Scatter(x=new_height_pnl['period'],
                                 y=new_height_pnl[on],
                                 mode='markers',
                                 name='new_max'))

        utils.update_legend_name(
            fig, dict(cum_return_dd='累计回报率回撤', new_max='累计回报率新高'))
        # styling
        fig.update_layout(styling.plot_layout)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    @param.depends('start_date', 'end_date', watch=True)
    def update(self):
        self.return_dd_plot.object = self.plot_drawdown()

    def __panel__(self):
        self._layout = pn.Column(
            pn.pane.HTML('<h1>回撤分析</h1>', sizing_mode='stretch_width'),
            self.date_range,
            self.return_dd_plot,
            sizing_mode='stretch_both',
            scroll=True,
            styles=self.styles,
        )
        return self._layout
