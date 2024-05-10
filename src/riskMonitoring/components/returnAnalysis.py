import riskMonitoring.processing as processing
import panel as pn
import pandas as pd
import plotly.express as px
from panel.viewable import Viewer
import param
import riskMonitoring.styling as styling
import riskMonitoring.utils as utils


class Component(Viewer):
    '''
    analysis on the return of the portfolio
    '''

    return_barplot = param.Parameterized()
    calculated_b_stock = param.Parameterized()
    calculated_p_stock = param.Parameterized()

    def __init__(self, styles, calculated_p_stock, calculated_b_stock, max_width, min_width, title, **params):
        self.styles = styles
        self.title = title
        self.max_width = max_width
        self.min_width = min_width
        self.calculated_p_stock = calculated_p_stock
        self.calculated_b_stock = calculated_b_stock

        start = self.calculated_p_stock.time.min().date()
        end = self.calculated_p_stock.time.max().date()

        self.range_slider = pn.widgets.DateRangeSlider(
            name='Date Range Slider',
            start=start, end=end,
            value=(start, end),
        )
        self.options_selector = pn.widgets.Select(
            name='选择周期',
            options=['每周回报', '每月回报', '每年回报'],
            value='每周回报'
        )

        self.return_barplot = pn.pane.Plotly()
        self.attribute_barplot = pn.pane.Plotly()

        self.update()

        super().__init__(**params)

    def _calculate_return(self, df, freq):

        df['period'] = df.time.dt.to_period(freq='D')
        # aggregate by date to have daily return, time keep the first entry of that day
        agg_df = df.groupby('period').agg(
            {'return': 'sum', 'time': 'first'}).reset_index()

        # aggregate by period
        agg_df['period'] = agg_df.time.dt.to_period(freq=freq)

        grouped = agg_df.groupby(['period'])

        agg_df['cum_return'] = grouped['return'].apply(
            lambda x: (x + 1).cumprod() - 1).reset_index(level=0, drop=True)

        agg_df = agg_df.loc[agg_df.groupby('period')['time'].idxmax()]
        return agg_df

    def update_aggregate_df(self, cliped_p, cliped_b, freq):
        p_return = self._calculate_return(cliped_p, freq)
        b_return = self._calculate_return(cliped_b, freq)

        merge_df = pd.merge(p_return, b_return, on='period',
                            how='outer', suffixes=('_p', '_b'))
        return merge_df

    def create_attributes_barplot(self, cliped_p, cliped_b, freq):
        attribute_df = self._update_attributes_df(cliped_p, cliped_b, freq)
        attribute_df['period'] = attribute_df['period'].dt.start_time.dt.strftime(
            '%Y-%m-%d')
        fig = px.bar(attribute_df, x='period', y=[
                     'allocation', 'selection', 'interaction', 'notional_active_return', 'active_return'])
        colname_to_name = {
            'allocation': '分配',
            'selection': '选择',
            'interaction': '交互',
            'notional_active_return': '名义主动回报',
            'active_return': '实际主动回报'
        }
        fig.for_each_trace(lambda t: t.update(name=colname_to_name.get(t.name, t.name),
                                              legendgroup=colname_to_name.get(
            t.name, t.name),
            hovertemplate=t.hovertemplate.replace(
            t.name, colname_to_name.get(t.name, t.name))
        ))

        fig.update_layout(barmode='group',
                          bargap=0.2, bargroupgap=0.0)
        fig.update_xaxes(type='category')
        fig.update_layout(**styling.plot_layout)
        fig.update_traces(**styling.barplot_trace)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    def create_return_barplot(self, cliped_p, cliped_b, freq):
        period_return = self.update_aggregate_df(cliped_p, cliped_b, freq)
        period_return.period = period_return.period.dt.start_time.dt\
            .strftime('%Y-%m-%d')
        fig = px.bar(period_return, x='period', y=[
                     'cum_return_b', 'cum_return_p'],
                     barmode='overlay',
                     )
        # update legend
        colname_to_name = {
            'cum_return_p': 'portfolio回报率',
            'cum_return_b': 'benchmark回报率'
        }
        fig.for_each_trace(lambda t: t.update(name=colname_to_name.get(t.name, t.name),
                                              legendgroup=colname_to_name.get(
                                                  t.name, t.name),
                                              hovertemplate=t.hovertemplate.replace(
                                                  t.name, colname_to_name.get(t.name, t.name))
                                              ))

        fig.update_layout(**styling.plot_layout)
        # prevent auto filling
        fig.update_xaxes(type='category')
        fig.update_traces(**styling.barplot_trace)
        fig.update_layout(legend_title_text=None)

        return fig.to_dict()

    @param.depends('calculated_p_stock', 'calculated_b_stock', 'options_selector.value', 'range_slider.value', watch=True)
    def update(self):
        freq = 'W-MON'
        if self.options_selector.value == "每日回报":
            freq = "D"
        elif self.options_selector.value == "每月回报":
            freq = 'M'
        elif self.options_selector.value == "每年回报":
            freq = 'Y'
        elif self.options_selector.value == "每周回报":
            freq = 'W-MON'

        cliped_p = utils.clip_df(
            start=self.range_slider.value[0], end=self.range_slider.value[1], df=self.calculated_p_stock)
        cliped_b = utils.clip_df(
            start=self.range_slider.value[0], end=self.range_slider.value[1], df=self.calculated_b_stock)

        # update return plot
        return_barplot = self.create_return_barplot(
            cliped_p=cliped_p, cliped_b=cliped_b, freq=freq)
        self.return_barplot.object = return_barplot

        # update attribute plot
        attributes_barplot = self.create_attributes_barplot(
            cliped_p=cliped_p, cliped_b=cliped_b, freq=freq)
        self.attribute_barplot.object = attributes_barplot

    def _update_attributes_df(self, cliped_p, cliped_b, freq):
        agg_p = processing.aggregate_analytic_df_by_period(cliped_p, freq)
        agg_b = processing.aggregate_analytic_df_by_period(cliped_b, freq)
        bhb_df = processing.calculate_periodic_BHB(agg_p, agg_b)
        agg_bhb = processing.aggregate_bhb_df(bhb_df)
        agg_bhb.reset_index(inplace=True)
        return agg_bhb

    def __panel__(self):
        self._layout = pn.Column(
            pn.pane.HTML(f'<h1>{self.title}</h1>'),
            self.range_slider,
            self.options_selector,
            pn.Column(pn.pane.HTML('<h3>周期回报率</h3>')),
            self.return_barplot,
            pn.Column(pn.pane.HTML('<h3>主动回报率归因</h3>')),
            self.attribute_barplot,
            sizing_mode='stretch_both',
            scroll=True,
            styles=self.styles,
        )

        return self._layout
