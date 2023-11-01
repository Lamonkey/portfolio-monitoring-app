import riskMonitoring.processing as processing
from datetime import timedelta
import panel as pn
from panel.viewable import Viewer
import param
from riskMonitoring import utils


class Component(Viewer):
    '''
    Display a tabulator to display best and worst stocks in a time window
    '''
    start_date = param.Parameter()
    end_date = param.Parameter()
    hidden_col = [
        'index',
        'open',
        'high',
        'low',
        'volume',
        'money',
        'pct',
        'sector',
        'ave_price',
        'weight',
        'ini_w',
        'name',
        'pnl',
        'rest_cap',
        'cum_return',
        'handling_fee',
        'return',
        'period',
    ]
    forzen_columns = ['time', 'display_name', 'cum_pnl', 'shares', 'period']
    tooltip = "在一个时间窗口中累计盈利最高和最低的股票，包括已经卖出的股票，如果表格的日期小于窗口的结束时间代表已经卖出"

    def create_tabulator(self):
        col_title_map = {
            'display_name': '股票名称',
            'ticker': '股票代码',
            'time': '日期',
            'return': '回报率',
            'cum_return': '累计回报率',
            'sector': '行业',
            'shares': '持仓',
            'cash': '价值',
            'cum_pnl': '累计盈利',
            'close': '收盘价',
            'aggregate_sector': "行业板块",
        }
        return pn.widgets.Tabulator(sizing_mode='stretch_width',
                                    layout='fit_data_stretch',
                                    hidden_columns=self.hidden_col,
                                    frozen_columns=self.forzen_columns,
                                    titles=col_title_map
                                    )

    # def _get_cum_return(self, df):
    #     '''return a df contain cumulative return at the end date'''
    #     result_df = processing.calculate_cum_return_rate(df=df,
    #                                             start=self.start_date,
    #                                             end=self.end_date)
    #     grouped = result_df.groupby('ticker')
    #     last_row = result_df.loc[grouped.time.idxmax()]
    #     return last_row

    def get_processed_df(self):
        '''
        calculate attributes and return a sorted dataframe on weighted return
        '''
        start = self._date_range.value[0]
        end = self._date_range.value[1]
        selected_df = utils.clip_df(df=self.analytic_df,
                                    start=start,
                                    end=end)
        # remove row has empty ticker
        selected_df = selected_df[selected_df.ticker != ''].copy()
        df = processing.calculate_cum_pnl(selected_df)
        # df = self._get_cum_return(df)
        last_row = df.loc[df.groupby('ticker').time.idxmax()]
        return last_row.sort_values(by='cum_pnl', ascending=False)

    @param.depends('start_date', 'end_date', watch=True)
    def update(self):
        result_df = self.get_processed_df()
        self.best_5_tabulator.value = result_df.head(5)
        self.worst_5_tabulator.value = result_df.tail(5)

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

        self.best_5_tabulator = self.create_tabulator()
        self.worst_5_tabulator = self.create_tabulator()
        self.update()
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
            pn.pane.Str('总pnl最高回报5只股票'),
            self.best_5_tabulator,
            pn.pane.Str('总pnl最低回报5只股票'),
            self.worst_5_tabulator,
            sizing_mode='stretch_both',
            scroll=True,
            styles=self.styles,

        )
        return self._layout
