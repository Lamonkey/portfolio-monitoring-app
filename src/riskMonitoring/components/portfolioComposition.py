import riskMonitoring.processing as processing
from datetime import datetime, time
import panel as pn
import plotly.express as px
import numpy as np
from panel.viewable import Viewer
import param
import riskMonitoring.styling as styling
import riskMonitoring.utils as utils


class Component(Viewer):
    '''
    component to view allocation of money in secter and stocks
    '''
    p_stock_df = param.Parameterized()

    def __init__(self, styles, analytic_df, max_width, min_width, **params):
        self.p_stock_df = analytic_df
        self.styles = styles

        # calculate daily capital
        processing.calculate_cum_return(self.p_stock_df)
        self.daily_cap_df = self._calculate_daily_total_capital(
            self.p_stock_df)

        self.date_slider = \
            pn.widgets.DateSlider(name='选择某日资金分布',
                                  start=self.p_stock_df.time.min(),
                                  end=self.p_stock_df.time.max(),
                                  value=self.p_stock_df.time.max(),
                                  )
        self.date_range = \
            pn.widgets.DateRangeSlider(name='选择资金分布走势区间',
                                       start=self.p_stock_df.time.min(),
                                       end=self.p_stock_df.time.max(),
                                       value=(self.p_stock_df.time.min(
                                       ), self.p_stock_df.time.max()),
                                       )
        self.tree_plot = pn.pane.Plotly()
        self.trend_plot = pn.pane.Plotly()
        self.stock_tabulator = pn.widgets.Tabulator(
            layout="fit_data_stretch", width_policy='max')

        self.file_name, self.download_button = self.stock_tabulator\
            .download_menu(
                text_kwargs={'name': 'Enter filename', 'value': 'default.csv'},
                button_kwargs={'name': 'Download table'}
            )

        self.update_treeplot_and_tabulator()
        self.update_trend_plot()
        super().__init__(**params)

    def _calculate_daily_total_capital(self, df):
        '''
        return daily total capital and daily market value
        '''
        agg_df = df.groupby('time').agg(
            {'return': 'sum', 'rest_cap': 'first', 'pnl': 'sum', 'cash': 'sum'})
        agg_df.reset_index(inplace=True)

        agg_df['rest_cap'] = agg_df['rest_cap'].fillna(method='ffill')

        return agg_df

    @param.depends('date_range.value', watch=True)
    def update_trend_plot(self):
        '''
        update the view of trend plot
        '''
        df = utils.clip_df(
            start=self.date_range.value[0],
            end=self.date_range.value[1],
            df=self.daily_cap_df)
        self.trend_plot.object = self.create_trend_plot(df)

    def create_trend_plot(self, df):
        fig = px.bar(df, x='time', y=['cash', 'rest_cap'])
        fig.update_layout(**styling.plot_layout)
        fig.update_traces(
            marker_line_width=0,
            selector=dict(type="bar"))
        fig.update_layout(bargap=0,
                          bargroupgap=0,
                          )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                          yaxis_title=None, xaxis_title=None,
                          margin=dict(l=0, r=0, t=0, b=0))
        return fig.to_dict()

    def create_treemap(self, cap_on_date, selected_df):
        fig = px.treemap(selected_df,
                         #  path=[px.Constant('cash_position'), 'position',
                         #        'aggregate_sector', 'display_name'],
                         path=['aggregate_sector', 'display_name'],
                         values='cash',
                         color='cum_return',
                         hover_data=['cum_return', 'pnl', 'cash', 'weight'],
                         color_continuous_scale='RdBu',
                         color_continuous_midpoint=np.average(
                             selected_df['cum_return'])
                         )
        fig.update_coloraxes(colorbar=dict(orientation='h', yanchor='bottom'))
        fig.update_layout(styling.plot_layout)
        fig.update_layout(coloraxis_colorbar=dict(
            title="累计加权回报率"))

        return fig.to_dict()

    def __panel__(self):

        self._layout = pn.Column(
            pn.pane.HTML('<h1>Portfolio 组成</h1>'),
            self.date_slider,
            pn.Tabs(('每日权重', self.tree_plot),
                    ('每日portfolio detail', pn.Column(self.stock_tabulator,
                                                     self.file_name,
                                                     self.download_button))),
            self.date_range,
            self.trend_plot,
            sizing_mode='stretch_both',
            styles=self.styles,
            scroll=True
        )
        return self._layout

    @param.depends('date_slider.value', watch=True)
    def update_treeplot_and_tabulator(self):
        # create datetime at 3pm, which is the at market close
        my_time = time(hour=15, minute=0, second=0)
        date_time = datetime.combine(self.date_slider.value, my_time)
        # total cap of that day
        cap_on_date = self.daily_cap_df[self.daily_cap_df.time == date_time]
        # cap of each ticker
        selected_df = self.p_stock_df[self.p_stock_df.time == date_time].copy()
        tree_plot = self.create_treemap(cap_on_date, selected_df)
        self.tree_plot.object = tree_plot


        # update tabulator
        '''
        ['time', 'ticker', 'open', 'close', 'high', 'low', 'volume', 'money',
       'shares', 'sector', 'aggregate_sector', 'display_name', 'name', 'cash',
       'rest_cap', 'ini_w', 'ave_price', 'handling_fee', 'weight', 'pct',
       'return', 'pnl', 'cum_return'],
        '''
        self.stock_tabulator.value = selected_df[[
            'display_name',
            'ticker',
            'close',
            'shares',
            'weight',
            'pnl',
            'cash',
            'rest_cap',
            'return',
            'cum_return',
            'time']]
