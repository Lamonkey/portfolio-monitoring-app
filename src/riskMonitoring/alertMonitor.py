from riskMonitoring import db_operation as db
class PNLDown():
    def watch(self):
        pass

    def trigger(self):
        pass


class MaxDrawnDown():
    def watch(self):
        pass

    def trigger(self):
        pass

class IsPonshuOkay():
    def watch():
        pass

class IsShuWongOkay():
    def watch():
        pass


class StockPriceDailyPct():
    '''
    return any latest stock in portfolio has a daily change over threshold

    Parameters
    ----------
    threshold: float
        threshold to trigger alert, represent percentage 
    '''

    def __init__(self, **parm):
        self.latest_analytic_p = db.get_most_recent_analytic_p()
        self.threshold = parm['threshold']
    
    def run(self):
        exceed = self.latest_analytic_p[self.latest_analytic_p['pct'] > self.threshold]
        return exceed

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
        self.analytic_p = self.analytic_p[self.analytic_p.time == self.analytic_p.time.max()]
        self.threshold = param['threshold']


    def run(self):
        result = self.portfolio_profile.merge(self.analytic_p[['ticker', 'close']], on='ticker', suffixes=('_profile', '_analytic'))
        result['cum_pct'] = result['close'] / result['ave_price'] - 1
        exceed = result[result['cum_pct'] > self.threshold]

        return exceed