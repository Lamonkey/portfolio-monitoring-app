import riskMonitoring.processing as processing
import panel as pn
import pandas as pd
import plotly.express as px
from panel.viewable import Viewer
import param
import riskMonitoring.styling as styling
import riskMonitoring.utils as utils


class Component(Viewer):

    start_date = param.Parameter()
    end_date = param.Parameter()

    def __init__(self, analytic_p: pd.DataFrame, analytic_b: pd.DataFrame, title, styles, **params):
        self.analytic_p = analytic_p
        self.analytic_b = analytic_b
        self.title = title
        self.styles = styles
        end_date = analytic_p.time.max().date()
        start_date = analytic_p.time.min().date()
        self.date_range = pn.widgets.DateRangeSlider(
            start=start_date,
            end=end_date,
            value=(end_date - pd.Timedelta(days=7), end_date)
        )
        self._sync_widgets()
        merged_df = self.processing()
        self.aw_plot = pn.pane.Plotly(self.plot_active_weight(merged_df))
        self.ar_plot = pn.pane.Plotly(self.plot_active_return(merged_df))
        self.cr_plot = pn.pane.Plotly(self.plot_cum_return(merged_df))
        self.oppo_cost_report = pn.pane.HTML(
            self.get_oppo_cost_report(merged_df), sizing_mode='stretch_width')
        super().__init__(**params)

    @param.depends('date_range.value', watch=True)
    def _sync_widgets(self):
        self.start_date = self.date_range.value[0]
        self.end_date = self.date_range.value[1]

    def get_oppo_cost_report(self, merged_df):
        oppo_save_pnl = merged_df['pnl']\
            .mul(merged_df['weight_p'].sub(merged_df['weight_b']))\
            .sum()
        merged_df['oppo_save_return'] = merged_df['return_p'] * \
            merged_df['active_weight']
        oppo_save_return = merged_df.groupby('aggregate_sector')[
            'oppo_save_return'].sum()
        html = f'<div style="display:flex; justify-content: space-between; align-items: center">\
            <h1>allocation\'s <br/>oppo saving: </h1> <h1 style>{round(oppo_save_pnl.sum(),2)}<br/> {round(oppo_save_return.sum() * 100, 2)}%</h1></div>'
        html += '<table style="width:100%; text-align:center">'
        html += '<tr><th>Sector</th><th>Value</th></tr>'

        for key, value in oppo_save_return.items():
            color = 'red' if value < 0 else 'green'
            html += f'<tr style="color: {color};"><td>{key}</td><td>{round(value * 100,2)}%</td></tr>'

        html += '</table>'
        return html

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
        self.oppo_cost_report.object = self.get_oppo_cost_report(merged_df)

    def processing(self):
        '''
        aggregate return by sector 
        '''
        # clipe by start and end date
        cliped_p = utils.clip_df(
            start=self.start_date, end=self.end_date, df=self.analytic_p)
        cliped_b = utils.clip_df(
            start=self.start_date, end=self.end_date, df=self.analytic_b)
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
                ('主动权重', pn.Column(self.oppo_cost_report, self.aw_plot)),
                ('主动回报', self.ar_plot),
                ('累计回报', self.cr_plot),
                dynamic=True,
            ),
            styles=self.styles,
            sizing_mode='stretch_both',
        )
        return self._laytout
