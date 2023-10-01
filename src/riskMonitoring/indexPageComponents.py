import processing
from datetime import datetime, timedelta, time
import panel as pn
import pandas as pd
import hvplot.pandas  # noqa
import plotly.express as px
import numpy as np
import hvplot.pandas  # noqa
from panel.viewable import Viewer
import param
import styling as styling
import description
import plotly.graph_objs as go
import utils as utils

# import warnings
pn.extension('mathjax')
pn.extension('plotly')


class SectorPerformance(Viewer):

    start_date = param.Parameter()
    end_date = param.Parameter()

    def __init__(self, analytic_p: pd.DataFrame, analytic_b: pd.DataFrame, title, styles, **params):
        self.analytic_p = analytic_p
        self.analytic_b = analytic_b
        self.title = title
        self.styles = styles
        self.date_range = pn.widgets.DateRangeSlider(
            start=analytic_p.time.min(),
            end=analytic_p.time.max(),
            value=(analytic_p.time.min(),
                   analytic_p.time.max())
        )
        self._sync_widgets()
        merged_df = self.processing()
        self.aw_plot = pn.pane.Plotly(self.plot_active_weight(merged_df))
        self.ar_plot = pn.pane.Plotly(self.plot_active_return(merged_df))
        self.cr_plot = pn.pane.Plotly(self.plot_cum_return(merged_df))
        super().__init__(**params)

    @param.depends('date_range.value', watch=True)
    def _sync_widgets(self):
        self.start_date = self.date_range.value[0]
        self.end_date = self.date_range.value[1]

    def plot_active_weight(self, merged_df):
        fig = px.bar(merged_df, x='period',
                     y='active_weight', color='aggregate_sector',
                     barmode='group'
                     )
        fig.update_layout(**styling.plot_layout)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    def plot_active_return(self, merged_df):
        fig = px.bar(merged_df, x='period',
                     y='active_return', color='aggregate_sector',
                     barmode='group')
        fig.update_layout(**styling.plot_layout)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    def plot_cum_return(self, merged_df):
        fig = px.line(merged_df, x='period', y='cum_return_p',
                      color='aggregate_sector')
        fig.update_layout(**styling.plot_layout)
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()

    @param.depends('start_date', 'end_date', watch=True)
    def update(self):
        '''
        trigger by date range changes
        '''
        merged_df = self.processing()
        self.aw_plot.object = self.plot_active_weight(merged_df)
        self.ar_plot.object = self.plot_active_return(merged_df)
        self.cr_plot.object = self.plot_cum_return(merged_df)

    def processing(self):
        '''
        aggregate return by sector 
        '''
        # clipe by start and end date
        cliped_p = utils.clip_df(
            self.start_date, self.end_date, self.analytic_p)
        cliped_b = utils.clip_df(
            self.start_date, self.end_date, self.analytic_b)
        # calculate cum_return for each sector
        agg_p = processing.agg_to_daily_sector(cliped_p)
        agg_b = processing.agg_to_daily_sector(cliped_b)

        # merge benchmark and portfolio
        agg_p['in_portfolio'] = True
        agg_b['in_benchmark'] = True
        merged_df = pd.merge(agg_p, agg_b, on=['period', 'aggregate_sector'],
                             how='outer', suffixes=('_p', '_b'))
        processing.post_process_merged_analytic_df(merged_df=merged_df)

        # calculate cumulative return
        merged_df['cum_return_p'] = merged_df\
            .groupby('aggregate_sector')['return_p']\
            .apply(lambda x: (x + 1).cumprod()-1)\
            .reset_index(level=0, drop=True)

        merged_df['cum_return_b'] = merged_df\
            .groupby('aggregate_sector')['return_b']\
            .apply(lambda x: (x + 1).cumprod()-1)\
            .reset_index(level=0, drop=True)

        # activate weight
        merged_df['weight_p'] = merged_df['weight_p'].fillna(0)
        merged_df['weight_b'] = merged_df['weight_b'].fillna(0)
        merged_df['active_weight'] = merged_df['weight_p'] - \
            merged_df['weight_b']
        merged_df['active_return'] = merged_df['return_p'] - \
            merged_df['return_b']

        # convert period to str
        merged_df['period'] = merged_df['period'].dt.strftime('%Y-%m-%d')
        return merged_df

    def __panel__(self):
        self._laytout = pn.Column(
            pn.pane.HTML(f'<h1>{self.title}</h1>'),
            self.date_range,
            pn.Tabs(
                ('主动权重', self.aw_plot),
                ('主动回报', self.ar_plot),
                ('累计回报', self.cr_plot),
                dynamic=True,
            ),
            styles=self.styles,
            sizing_mode='stretch_both',
        )
        return self._laytout


class TrendPlot(Viewer):
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


class OverviewCard(Viewer):
    '''
    summary on the portfolio performance vs benchmark performance
    '''

    start_date = param.Parameter()
    end_date = param.Parameter()
    b_stock_df = param.Parameter()
    p_stock_df = param.Parameter()

    def format_number(self, num):
        return f'{round(num * 100, 2)}%'

    def get_color(self, num):
        return 'green' if num >= 0 else 'red'

    def create_report(self, df):
        '''
        create a html report
        '''
        # Calculate the risk, tracking error, active return

        # use accumulative result from last row
        most_recent_row = df.loc[df.period.idxmax(
        )]
        active_return = most_recent_row.cum_return_p - most_recent_row.cum_return_b
        tracking_error = most_recent_row.tracking_error
        total_return = most_recent_row.cum_return_p
        cum_pnl = most_recent_row.cum_pnl
        risk = most_recent_row.risk
        
        # calculate attributes for active return
        start = self.date_range_slider.value[0]
        end = self.date_range_slider.value[1]
        first_last_p = processing.select_first_last_stock_within_window(
            self.p_stock_df, start, end)
        first_last_b = processing.select_first_last_stock_within_window(
            self.b_stock_df, start, end)

        # pct (unweighted return)
        first_last_p['pct'] = first_last_p.groupby('ticker').close.pct_change()
        first_last_b['pct'] = first_last_b.groupby('ticker').close.pct_change()

        # use first row as previous weight
        first_last_p['prev_weight'] = first_last_p.groupby(
            'ticker').weight.shift(1)
        first_last_b['prev_weight'] = first_last_b.groupby(
            'ticker').weight.shift(1)

        # keep only last entry for each ticker
        last_p = first_last_p.dropna(subset=['pct'])
        last_b = first_last_b.dropna(subset=['pct'])

        # combine for calculation
        last_p['in_portfolio'] = True
        last_b['in_benchmark'] = True
        merged_df = pd.merge(last_p, last_b, on='ticker',
                             how='outer', suffixes=('_p', '_b'))
        processing.post_process_merged_analytic_df(merged_df=merged_df)

        # fill empty weight and pct with 0
        merged_df['prev_weight_p'] = merged_df['prev_weight_p'].fillna(0)
        merged_df['prev_weight_b'] = merged_df['prev_weight_b'].fillna(0)
        merged_df['pct_p'] = merged_df['pct_p'].fillna(0)
        merged_df['pct_b'] = merged_df['pct_b'].fillna(0)
        
        # allocation, interaction, selection and notional active return
        merged_df['allocation'] = (
            merged_df.prev_weight_p - merged_df.prev_weight_b) * merged_df.pct_b
        merged_df['interaction'] = (
            merged_df.pct_p - merged_df.pct_b) * (merged_df.prev_weight_p - merged_df.prev_weight_b)
        merged_df['selection'] = (
            merged_df.pct_p - merged_df.pct_b) * merged_df.prev_weight_b
        merged_df['notional_active_return'] = merged_df['allocation'] + \
            merged_df['interaction'] + merged_df['selection']

        notional_return = merged_df.notional_active_return.sum()
        interaction = merged_df.interaction.sum()
        allocation = merged_df.allocation.sum()
        selection = merged_df.selection.sum()

        # Create a function for text report
        report = f"""
<style>
    .compact-container {{
        display: flex;
        flex-direction: column;
        gap: 5px;
    }}
    
    .compact-container > div {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 2px;
    }}
    
    .compact-container > div > h2,
    .compact-container > div > h3,
    .compact-container > div > p,
    .compact-container > div > ul > li {{
        margin: 0;
    }}
    
    .compact-container > ul {{
        padding: 0;
        margin: 0;
        list-style-type: none;
    }}
    
    .compact-container > ul > li {{
        display: flex;
        margin-bottom: 2px;
    }}
</style>

<div class="compact-container">
    <u><b>总收益</b></u>
    <div>
        <h2 style="margin: 0;">¥{round(cum_pnl,2)}</h2>
        <h2 style='color: {self.get_color(total_return)}; margin: 0;'>{self.format_number(total_return)}</h2>
    </div>
    <div>
        <p style="margin: 0;">追踪误差</p>
        <p style='color: {self.get_color(tracking_error)}; margin: 0;'>{self.format_number(tracking_error)}</p>
    </div>
    <div>
        <p style="margin: 0;">风险</p>
        <p style='color: {self.get_color(risk)}; margin: 0;'>{self.format_number(risk)}</p>
    </div>
    <div>
        <p style="margin: 0;">主动回报归因</p>
        <ul style="padding: 0; margin: 0; list-style-type: none;">
            <li style="margin-bottom: 2px;">
                <div style="display: flex;">
                    <p style="margin: 0;">实际主动回报:</p>
                    <p style="color: {self.get_color(active_return)}; margin: 0;">{self.format_number(active_return)}</p>
                </div>
            </li>
            <li style="margin-bottom: 2px;">
                <div style="display: flex;">
                    <p style="margin: 0;">名义主动回报:</p>
                    <p style="color: {self.get_color(notional_return)}; margin: 0;">{self.format_number(notional_return)}</p>
                </div>
            </li>
            <li style="margin-bottom: 2px;">
                <div style="display: flex;">
                    <p style="margin: 0;">分配:</p>
                    <p style="color: {self.get_color(allocation)}; margin: 0;">{self.format_number(allocation)}</p>
                </div>
            </li>
            <li style="margin-bottom: 2px;">
                <div style="display: flex;">
                    <p style="margin: 0;">选择:</p>
                    <p style="color: {self.get_color(selection)}; margin: 0;">{self.format_number(selection)}</p>
                </div>
            </li>
            <li style="margin-bottom: 2px;">
                <div style="display: flex;">
                    <p style="margin: 0;">交互:</p>
                    <p style="color: {self.get_color(interaction)}; margin: 0;">{self.format_number(interaction)}</p>
                </div>
            </li>
        </ul>
    </div>
</div>
"""

        return report

    def create_cum_pnl_plot(self, df):
        fig = px.line(df, x='x', y='cum_pnl')
        fig.update_layout(styling.plot_layout)
        return fig.to_dict()

    def create_tracking_error_plot(self):
        pass

    def create_risk_plot(self):
        pass


    def create_return_ratio(self, df):
        df['cum_return_ratio'] = df['cum_return_p'] / df['cum_return_b']
        fig = px.line(df, x='x', y='cum_return_ratio')
        fig.update_traces(mode="lines+markers",
                          marker=dict(size=5), line=dict(width=2))
        fig.update_layout(styling.plot_layout)
        colname_to_name = {
            'cum_return_ratio': 'portfolio累计回报率/benchmark累计回报率'
        }
        fig.for_each_trace(lambda t: t.update(name=colname_to_name.get(t.name, t.name),
                                              legendgroup=colname_to_name.get(
            t.name, t.name),
            hovertemplate=t.hovertemplate.replace(
            t.name, colname_to_name.get(t.name, t.name))
        ))
       
        return fig.to_dict()

    def create_cum_return_plot(self, df):

        fig = px.line(df, x='x', y=[
                      'cum_return_p', 'cum_return_b'])
        fig.update_traces(mode="lines+markers",
                          marker=dict(size=5), line=dict(width=2))
        fig.update_layout(styling.plot_layout)
        colname_to_name = {
            'cum_return_p': 'Portfolio累计回报率',
            'cum_return_b': 'benchmark累计回报率'
        }
        fig.for_each_trace(lambda t: t.update(name=colname_to_name.get(t.name, t.name),
                                              legendgroup=colname_to_name.get(
            t.name, t.name),
            hovertemplate=t.hovertemplate.replace(
            t.name, colname_to_name.get(t.name, t.name))
        ))
        fig.update_layout(legend_title_text=None)
        return fig.to_dict()


    @param.depends('date_range_slider.value', 'b_stock_df', 'p_stock_df', watch=True)
    def update(self):
        start = self.date_range_slider.value[0]
        end = self.date_range_slider.value[1]
        clip_p = utils.clip_df(start, end, self.p_stock_df)
        clip_b = utils.clip_df(start, end, self.b_stock_df)
        df = processing.get_portfolio_anlaysis(analytic_b=clip_b, analytic_p=clip_p)
        df['x'] = df['period'].dt.start_time.dt.strftime('%Y-%m-%d')
        self.report.object = self.create_report(df)
        self.return_plot.object = self.create_cum_return_plot(df)
        self.ratio_plot.object = self.create_return_ratio(df)
        self.cum_pnl_plot.object = self.create_cum_pnl_plot(df)

    def __init__(self, b_stock_df, p_stock_df, styles, **params):
        self.styles = styles
        self.b_stock_df = b_stock_df
        self.p_stock_df = p_stock_df
        self.date_range_slider = pn.widgets.DateRangeSlider(
            start=p_stock_df.time.min(),
            end=b_stock_df.time.max(),
            value=(p_stock_df.time.max() -
                   timedelta(days=7), p_stock_df.time.max())
        )
        self.cum_pnl_plot = pn.pane.Plotly()
        self.return_plot = pn.pane.Plotly()
        self.ratio_plot = pn.pane.Plotly()
        self.report = pn.pane.HTML(sizing_mode='stretch_width')
        self.update()

        super().__init__(**params)
        # self._sync_widgets()

    def __panel__(self):
        self._layout = pn.Column(
            pn.Column(
                pn.pane.HTML('<h1>投资组合总结</h1>'),
                # pn.widgets.TooltipIcon(value='a simple tootip <br> switch to anotherline')
            ),
            self.date_range_slider,
            self.report,
            pn.Tabs(
                ("累计收益", self.cum_pnl_plot),
                ('累计回报率', self.return_plot),
                ('累计回报率比例', self.ratio_plot),

            ),
            sizing_mode='stretch_both',
            styles=self.styles,
            scroll=True,
        )
        return self._layout

    # @param.depends('value', 'width', watch=True)
    # def _sync_widgets(self):
    #     pass

    @param.depends('_date_range.value', watch=True)
    def _sync_params(self):
        self.start_date = self.date_range_slider.value[0]
        self.end_date = self.date_range_slider.value[1]


class DrawDownCard(Viewer):
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

        utils.update_legend_name(fig, dict(cum_return_dd='累计回报率回撤',new_max='累计回报率新高'))
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


class ReturnAnlaysisCard(Viewer):
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

        self.range_slider = pn.widgets.DateRangeSlider(
            name='Date Range Slider',
            start=self.calculated_p_stock.time.min(), end=self.calculated_p_stock.time.max(),
            value=(self.calculated_p_stock.time.min(),
                   self.calculated_p_stock.time.max()),

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
            self.range_slider.value[0], self.range_slider.value[1], self.calculated_p_stock)
        cliped_b = utils.clip_df(
            self.range_slider.value[0], self.range_slider.value[1], self.calculated_b_stock)

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


class PortfolioCompositionCard(Viewer):
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

        self.date_slider = pn.widgets.DateSlider(name='选择某日资金分布',
                                                 start=self.p_stock_df.time.min(),
                                                 end=self.p_stock_df.time.max(),
                                                 value=self.p_stock_df.time.max(),
                                                 )
        self.date_range = pn.widgets.DateRangeSlider(name='选择资金分布走势区间',
                                                     start=self.p_stock_df.time.min(),
                                                     end=self.p_stock_df.time.max(),
                                                     value=(self.p_stock_df.time.min(
                                                     ), self.p_stock_df.time.max()),
                                                     )
        self.tree_plot = pn.pane.Plotly()
        self.trend_plot = pn.pane.Plotly()
        self.update_treeplot()
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
        # for index, _ in agg_df.iterrows():
        #     if index == 0:
        #         continue
        #     previous_total_capital = agg_df.at[index - 1, 'total_capital']
        #     agg_df.loc[index, 'total_capital'] = agg_df.loc[index,
        #                                                     'pnl'] + previous_total_capital
        return agg_df

    @param.depends('date_range.value', watch=True)
    def update_trend_plot(self):
        '''
        update the view of trend plot
        '''
        df = utils.clip_df(
            self.date_range.value[0], self.date_range.value[1], self.daily_cap_df)
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
        # idle_cap = cap_on_date['rest_cap'] - cap_on_date['cash']
        # selected_df['position'] = '股票'
        # not_in_portfolio_row = pd.DataFrame({
        #     'display_name': ['闲置'],
        #     'position': ['闲置'],
        #     'aggregate_sector': ['闲置'],
        #     'cash': [idle_cap.values[0]],
        #     'cum_return': [0]
        # })
        # df = pd.concat([selected_df, not_in_portfolio_row],
        #                ignore_index=True)

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
            pn.pane.HTML('<h1>资金分布</h1>'),
            self.date_slider,
            self.tree_plot,
            self.date_range,
            self.trend_plot,
            sizing_mode='stretch_both',
            styles=self.styles,
            scroll=True
        )
        # self._layout = pn.Card(self.datetime_picker,
        #                        self.tree_plot,
        #                        self.date_range,
        #                        self.trend_plot,
        #                        max_width=self.max_width,
        #                        min_width=self.min_width,
        #                        styles=styling.card_style,
        #                        header=pn.pane.Str('资金分布'))
        return self._layout

    @param.depends('date_slider.value', watch=True)
    def update_treeplot(self):
        # add midnight time, becasue all daily price's time is midnight
        date_time = datetime.combine(self.date_slider.value, time.min)
        # total cap of that day
        cap_on_date = self.daily_cap_df[self.daily_cap_df.time == date_time]
        # cap of each ticker
        selected_df = self.p_stock_df[self.p_stock_df.time == date_time].copy()
        tree_plot = self.create_treemap(cap_on_date, selected_df)
        self.tree_plot.object = tree_plot


class BestAndWorstStocks(Viewer):
    start_date = param.Parameter()
    end_date = param.Parameter()
    hidden_col = [
        'index',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'money',
        'pct',
        'sector',
        'aggregate_sector',
        'ave_price',
        'weight',
        'ini_w',
        'name',
        'pnl',
        'rest_cap'
    ]
    forzen_columns = ['display_name', 'cum_return', 'cum_pnl', 'shares']
    tooltip = "在一个时间窗口中累计盈利最高和最低的股票，包括已经卖出的股票，如果表格的日期小于窗口的结束时间代表已经卖出"

    def create_tabulator(self, df):
        col_title_map = {
            'display_name': '股票名称',
            'ticker': '股票代码',
            'time': '日期',
            'return': '回报率',
            'cum_return': '累计回报率',
            'sector': '行业',
            'shares': '持仓',
            'cash': '现金',
            'cum_pnl': '累计盈利',
        }
        return pn.widgets.Tabulator(df, sizing_mode='stretch_width',
                                    hidden_columns=self.hidden_col,
                                    frozen_columns=self.forzen_columns,
                                    titles=col_title_map
                                    )

    @param.depends('start_date', 'end_date', watch=True)
    def update(self):
        result_df = self.get_processed_df()
        self.best_5_tabulator.value = result_df.head(5)
        self.worst_5_tabulator.value = result_df.tail(5)

    def _get_cum_return(self, df):
        '''return a df contain cumulative return at the end date'''
        result_df = processing.calcualte_return(df=df,
                                                start=self.start_date,
                                                end=self.end_date)
        grouped = result_df.groupby('ticker')
        last_row = result_df.loc[grouped.time.idxmax()]
        return last_row

    def get_processed_df(self):
        '''
        calculate attributes and return a sorted dataframe on weighted return
        '''
        df = processing.calculate_cum_pnl(self.analytic_df,
                                          start=self.start_date,
                                          end=self.end_date)
        df = self._get_cum_return(df)
        return df.sort_values(by='cum_pnl', ascending=False)

    def __init__(self, analytic_df, styles, title, **params):
        self.styles = styles
        self.title = title
        self.analytic_df = analytic_df
        self._date_range = pn.widgets.DateRangeSlider(
            name='选择计算回报的时间区间',
            start=self.analytic_df.time.min(),
            end=self.analytic_df.time.max(),
            value=(self.analytic_df.time.max() -
                   timedelta(days=7), self.analytic_df.time.max())
        )
        self.start_date = self._date_range.value_start
        self.end_date = self._date_range.value_end
        result_df = self.get_processed_df()
        self.best_5_tabulator = self.create_tabulator(result_df.head(5))
        self.worst_5_tabulator = self.create_tabulator(result_df.tail(5))
        super().__init__(**params)

    @param.depends('_date_range.value', watch=True)
    def _sync_params(self):
        self.start_date = self._date_range.value[0]
        self.end_date = self._date_range.value[1]
        # print('update range...')

    def __panel__(self):
        self._layout = pn.Column(
            pn.pane.HTML(f"<h1>{self.title}</h1>"),
            self._date_range,
            pn.pane.Str('加权回报率最高回报5只股票'),
            self.best_5_tabulator,
            pn.pane.Str('加权回报率最低回报5只股票'),
            self.worst_5_tabulator,
            sizing_mode='stretch_both',
            scroll=True,
            styles=self.styles,

        )
        return self._layout


class TopHeader(Viewer):
    '''
    display up to todays' PnL, total return and max drawdown
    '''
    eval_df = param.Parameter()

    @param.depends('eval_df', watch=True)
    def update(self):
        '''
        update Pnl, total return and max drawdown when df is updated
        '''
        return

    def _process(self):
        '''calculate accumulative pnl, total return and Max Drawdown on return'''

        # accumulative return
        agg_df = processing.get_draw_down(self.eval_df)

        # last row
        last_row = agg_df.loc[agg_df.period.idxmax()]

        return last_row.cum_pnl, last_row.cum_return, last_row.cum_return_dd

    def create_report(self, pnl, total_return, max_drawdown):
        return pn.FlexBox(
            f"PnL:{round(pnl,2)}¥", f"回报：{round(total_return * 100,2)}%", f'最大回撤:{round(max_drawdown * 100,2)}%', justify_content='space-evenly')

    def __init__(self, eval_df, **params):
        self.eval_df = eval_df
        cum_pnl, total_return, max_drawdown = self._process()
        self.report = self.create_report(cum_pnl, total_return, max_drawdown)
        super().__init__(**params)

    def __panel__(self):
        self._layout = pn.Column(
            self.report, sizing_mode='stretch_both', scroll=True)
        return self._layout
