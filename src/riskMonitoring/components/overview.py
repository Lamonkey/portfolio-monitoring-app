import riskMonitoring.processing as processing
from datetime import timedelta
import panel as pn
import pandas as pd
import plotly.express as px
from panel.viewable import Viewer
import param
import riskMonitoring.styling as styling
import riskMonitoring.utils as utils


class Component(Viewer):
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
        most_recent_row = df.loc[df.time.idxmax(
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
        fig.update_traces(styling.line_plot_trace)
        return fig.to_dict()

    def create_return_ratio(self, df):
        df['cum_return_ratio'] = df['cum_return_p'] / df['cum_return_b']
        fig = px.line(df, x='x', y='cum_return_ratio')
        fig.update_traces(styling.line_plot_trace)
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


    def create_risk_plot(self, df):
        fig = px.line(df.dropna(subset=['risk']), x='x', y=['risk', 'return_p'])
        fig.update_traces(mode="lines+markers",
                          marker=dict(size=5),
                          line=dict(width=2))
        fig.update_layout(styling.plot_layout)

        return fig.to_dict()

    def create_tracking_error_plot(self, df):
        fig = px.line(df.dropna(subset='tracking_error'), x='x', y=['tracking_error', 'active_return'])
        fig.update_traces(mode="lines+markers",
                          marker=dict(size=5),
                          line=dict(width=2))
        fig.update_layout(styling.plot_layout)
        return fig.to_dict()
    
    def create_raw_data_table(self, df):
        return df
                                
    @param.depends('date_range_slider.value', 'b_stock_df', 'p_stock_df', watch=True)
    def update(self):
        start = self.date_range_slider.value[0]
        end = self.date_range_slider.value[1]
        clip_p = utils.clip_df(start=start, end=end, df=self.p_stock_df)
        clip_b = utils.clip_df(start=start, end=end, df=self.b_stock_df)
        df = processing.get_portfolio_anlaysis(
            analytic_b=clip_b, analytic_p=clip_p)
        df['x'] = df['time']
        # df['x'] = df['period'].dt.start_time.dt.q   strftime('%Y-%m-%d')
        self.report.object = self.create_report(df)
        self.return_plot.object = self.create_cum_return_plot(df) 
        self.ratio_plot.object = self.create_return_ratio(df)
        self.cum_pnl_plot.object = self.create_cum_pnl_plot(df)
        self.risk_plot.object = self.create_risk_plot(df)
        self.tracking_error_plot.object = self.create_tracking_error_plot(df)
        self.result_table.value = self.create_raw_data_table(df)

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
        self.risk_plot = pn.pane.Plotly()
        self.tracking_error_plot = pn.pane.Plotly()
        self.result_table = pn.widgets.Tabulator(sizing_mode='stretch_both', pagination='local')
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
                ('风险', self.risk_plot),
                ('追踪误差', self.tracking_error_plot),
                ('原始数据', self.result_table),
                dynamic=True,

            ),
            sizing_mode='stretch_both',
            styles=self.styles,
            scroll=True,
        )
        return self._layout

    @param.depends('_date_range.value', watch=True)
    def _sync_params(self):
        self.start_date = self.date_range_slider.value[0]
        self.end_date = self.date_range_slider.value[1]
