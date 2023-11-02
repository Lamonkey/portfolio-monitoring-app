from riskMonitoring import db_operation as db
from riskMonitoring import processing
class MDD():
    '''
    alert when portfolio has max draw down over threshold
    '''
    def __init__(self, **param):
        self.threshold = param['threshold']
        self.title = f'⚠️ 最大回撤超过 {self.threshold:.2%}'
        self.columns = [
            'time_p',
            'total_cap',
            'total_cap_max_drawdown',
        ]
        self.col_2_name = {
            'time_p': '记录日期',
            'total_cap': '总资产',
            'total_cap_max_drawdown': '最大回撤',
        }

    def run(self):
        analytic_p = db.get_portfolio_analytic_df()
        analytic_b = db.get_portfolio_analytic_df()
        mdd_df = processing.get_draw_down(analytic_p, analytic_b)
        
        # keep row with max time
        self.result = mdd_df[mdd_df.period == mdd_df.period.max()]
        self.result = self.result[self.result.total_cap_max_drawdown < self.threshold]
        # check mdd of total capital exceed threshold
        return self.result['total_cap_max_drawdown'] < self.threshold

    def get_result(self):
        if self.result is None:
            raise ValueError('run method must be called first')
        formatted = self.result[self.columns].copy()
        formatted['total_cap_max_drawdown'] = formatted['total_cap_max_drawdown'].apply(
            lambda x: f'{x:.2%}')
        formatted['total_cap'] = formatted['total_cap'].apply(
            lambda x: f'{format(x, ",")} ¥')
        formatted = formatted.rename(columns=self.col_2_name)
        return formatted


class StockPriceDailyPct():
    '''
    return any latest stock in portfolio has a daily change over threshold

    Parameters
    ----------
    threshold: float
        threshold to trigger alert, represent percentage 
    '''

    def __init__(self, **parm):
        self.threshold = parm['threshold']
        self.title = f'⚠️ 股票日涨跌幅超过 {self.threshold:.2%}'
        self.columns = ['ticker','shares','time','close','pct','display_name', 'aggregate_sector']
        self.col_2_name = {
            'ticker': '股票代码',
            'shares': '持仓',
            'time': '记录日期',
            'close': '目前价格',
            'pct': '日涨跌幅',
            'display_name': '股票名称',
            'aggregate_sector': '行业板块'
        }
    
    def get_result(self):
        if self.result is None:
            raise ValueError('run method must be called first')
        formated_result = self.result[self.columns].copy()
        formated_result = formated_result.rename(columns=self.col_2_name)
        return formated_result

    def run(self):
        latest_analytic_p = db.get_most_recent_analytic_p()
        self.result = latest_analytic_p[latest_analytic_p['pct']
                                        > self.threshold]
        return len(self.result) > 0


class StockPriceCumChange():
    '''
    return any latest stock in portfolio has a accumulative change of average price over threshold

    Parameters
    ----------
    threshold: float
        threshold to trigger alert, represent percentage 
    '''

    def __init__(self, **param):
        self.analytic_p = db.get_portfolio_analytic_df()
        self.portfolio_profile = db.get_most_recent_portfolio_profile()
        self.analytic_p = self.analytic_p[self.analytic_p.time ==
                                          self.analytic_p.time.max()]
        self.threshold = param['threshold']
        self.title = f'⚠️ 股票累计涨跌幅超过 {self.threshold:.2%}'
        self.columns = ['ticker', 'shares', 'date', 'aggregate_sector',
                        'display_name', 'cash', 'weight', 'ave_price', 'close', 'cum_pct']
        
        self.col_2_name = {
            'ticker': '股票代码',
            'shares': '持仓',
            'date': '记录日期',
            'aggregate_sector': '行业板块',
            'display_name': '股票名称',
            'cash': '价值',
            'weight': '权重',
            'ave_price': '购入价格',
            'close': '目前价格',
            'cum_pct': '累计涨跌幅'
        }

    def get_result(self):
        if self.result is None:
            raise ValueError('run method must be called first')
        formated_result = self.result[self.columns].copy()
        formated_result = formated_result.rename(columns=self.col_2_name)
        return formated_result


    def run(self):
        profile = db.get_most_recent_portfolio_profile()
        analytic_p = db.get_most_recent_analytic_p()
        processed_df = profile.merge(
            analytic_p[['ticker', 'close']], on='ticker', suffixes=('_profile', '_analytic'))
        processed_df['cum_pct'] = processed_df['close'] / \
            processed_df['ave_price'] - 1
        self.result = processed_df[processed_df['cum_pct'] > self.threshold]
        return len(self.result) > 0
