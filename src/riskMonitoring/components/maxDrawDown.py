
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
        self.pnl_dd_plot = pn.pane.Plotly(self.plot_drawdown('cum_pnl_dd'))

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

        # process cum_pnl and cum_return to percentage with 2 decimal
        df['cum_pnl'] = df['cum_pnl'].apply(
            lambda x: round(x, 2))
        df['cum_return'] = df['cum_return'].apply(
            lambda x: round(x * 100, 2))

        df.period = df.period.dt.strftime('%Y-%m-%d')
        hover_data = 'cum_pnl' if on == 'cum_pnl_dd' else 'cum_return'
        ending_unit = "¥" if on == 'cum_pnl_dd' else "%"
        fig = px.line(df, x='period', y=[on], custom_data=[hover_data])

        fig.update_traces(styling.line_plot_trace)
        hover_title = "累计回报率" if on == 'cum_return_dd' else "累计pnl"
        fig.update_traces(
            hovertemplate="<br>".join([
                hover_title + ": %{customdata[0]}" + ending_unit
            ])
        )
        utils.update_legend_name(
            fig, dict(cum_return_dd='累计回报率最大回撤',
                      new_max='累计回报率新高',
                      cum_pnl_dd='累计pnl最大回撤',
                      cum_pnl='累计pnl'))
        # styling
        fig.update_layout(styling.plot_layout)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    @param.depends('start_date', 'end_date', watch=True)
    def update(self):
        self.return_dd_plot.object = self.plot_drawdown()
        self.pnl_dd_plot.object = self.plot_drawdown('cum_pnl_dd')

    def __panel__(self):
        self._layout = pn.Column(
            pn.pane.HTML('<h1>最大回撤</h1>', sizing_mode='stretch_width'),
            self.date_range,
            pn.Tabs(
                ('累计回报率', self.return_dd_plot),
                ('累计pnl', self.pnl_dd_plot),
                dynamic=True,
            ),
            sizing_mode='stretch_both',
            scroll=True,
            styles=self.styles,
        )
        return self._layout
