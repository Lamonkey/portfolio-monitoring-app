import panel as pn
import plotly.express as px
from panel.viewable import Viewer
import param
import riskMonitoring.styling as styling
import riskMonitoring.utils as utils
import riskMonitoring.processing as processing


class Component(Viewer):
    selected_key_column = param.Parameter()
    calcualted_p_stock = param.Parameter()
    start_date = param.Parameter()
    end_date = param.Parameter()

    def __init__(self, styles, calculated_p_stock, analytic_b, max_width, min_width, **params):
        '''
        max draw down on total_cap, cum_pnl and active return
        '''
        self.styles = styles
        self.max_width = max_width
        self.min_width = min_width
        start = calculated_p_stock.time.min().date()
        end = calculated_p_stock.time.max().date()
        self.date_range = pn.widgets.DateRangeSlider(
            start=start,
            end=end,
            value=(start, end)
        )

        self.calculated_p_stock = calculated_p_stock
        self.analytic_b = analytic_b
        self._sycn_params()
        self.captial_mdd_plot = pn.pane.Plotly(
            self.plot_drawdown('total_cap'))
        self.pnl_mdd_plot = pn.pane.Plotly(self.plot_drawdown('cum_pnl'))
        super().__init__(**params)

    @param.depends('date_range.value', watch=True)
    def _sycn_params(self):
        self.start_date = self.date_range.value[0]
        self.end_date = self.date_range.value[1]

    def plot_drawdown(self, on='total_cap'):
        '''
        calcualte and plot max draw down on selected column
        Parameters
        ----------
        on: str
            column name to calculate max draw down
        '''
        cliped_df_p = utils.clip_df(df=self.calculated_p_stock,
                                    start=self.start_date,
                                    end=self.end_date)

        cliped_df_b = utils.clip_df(
            df=self.analytic_b, start=self.start_date, end=self.end_date
        )

        df = processing.get_draw_down(
            analytic_p=cliped_df_p, analytic_b=cliped_df_b)

        # process cum_pnl and cum_return to percentage with 2 decimal
        # df['cum_pnl'] = df['cum_pnl'].apply(
        #     lambda x: round(x, 2))
        # df['total_cap'] = df['total_cap'].apply(
        #     lambda x: round(x, 2))
        # df['active_return'] = df['active_return'].apply(
        #     lambda x: round(x * 100, 2))

        df.period = df.period.dt.strftime('%Y-%m-%d')
        fig = px.line(df, x='period', y=[
                      f'{on}_max_drawdown', f'{on}_drawdown'], custom_data=[on, 'period'])
        fig.update_traces(styling.line_plot_trace)
        hover_title = "总资产回撤" if on == 'total_cap' else "累计盈利回撤"
        fig.update_traces(
            hovertemplate="<br>".join([
                hover_title + ": %{customdata[0]}" + "¥",
                "日期: %{customdata[1]}",
            ])
        )
        utils.update_legend_name(
            fig, dict(
                cum_pnl='累计pnl',
                total_cap='总资产',
                active_return='主动回报'))

        # styling
        fig.update_layout(styling.plot_layout)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    @param.depends('start_date', 'end_date', watch=True)
    def update(self):
        self.captial_mdd_plot.object = self.plot_drawdown('total_cap')
        self.pnl_mdd_plot.object = self.plot_drawdown('cum_pnl')

    def __panel__(self):
        self._layout = pn.Column(
            pn.pane.HTML('<h1>最大回撤</h1>', sizing_mode='stretch_width'),
            self.date_range,
            pn.Tabs(
                ('总资产', self.captial_mdd_plot),
                ('累计盈利', self.pnl_mdd_plot),
                dynamic=True,
            ),
            sizing_mode='stretch_both',
            scroll=True,
            styles=self.styles,
        )
        return self._layout
