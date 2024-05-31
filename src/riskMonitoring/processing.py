import pandas as pd
import math
from datetime import datetime
import hvplot.pandas
import math
import numpy as np
from riskMonitoring.settings import HANDLE_FEE


def get_processing_result_of_stocks_df(stock_df, profile_df):

    # add sector_name display_name name
    ticker_sector_map = dict(
        zip(profile_df['ticker'], profile_df['aggregate_sector']))
    ticker_display_name_map = dict(
        zip(profile_df['ticker'], profile_df['display_name']))
    ticker_name_map = dict(zip(profile_df['ticker'], profile_df['name']))
    stock_df['display_name'] = stock_df['ticker'].map(ticker_display_name_map)
    stock_df['name'] = stock_df['ticker'].map(ticker_name_map)
    stock_df['aggregate_sector'] = stock_df['ticker'].map(ticker_sector_map)

    # calculate pct using closing price
    stock_df.sort_values(by=['date'], inplace=True)
    stock_df['pct'] = stock_df.groupby('ticker')['close'].pct_change()

    # calculate weight TODO: think about how to optimize this
    stock_df = stock_df.merge(profile_df[['weight', 'date', 'ticker']], on=[
                              'ticker', 'date'], how='outer')
    stock_df.rename(columns={'weight': 'initial_weight'}, inplace=True)

    # create if not in stock_df
    stock_df['current_weight'] = float('nan')
    stock_df['previous_weight'] = float('nan')
    df_grouped = stock_df.groupby('ticker')
    for _, group in df_grouped:
        pre_w = float('nan')
        ini_w = float('nan')
        for index, row in group.iterrows():
            cur_w = float('nan')

            # if has initial weight, the following row all use this initial weight
            if not pd.isna(row['initial_weight']):
                ini_w = row['initial_weight']
                pre_w = ini_w
                cur_w = ini_w

            # just calculate current weight based on previous weight
            else:
                cur_w = pre_w * (1 + row['pct'])

            stock_df.loc[index, 'current_weight'] = cur_w
            stock_df.loc[index, 'previous_weight'] = pre_w
            stock_df.loc[index, 'initial_weight'] = ini_w
            pre_w = cur_w

    stock_df.rename(columns={'weight': 'initial_weight'}, inplace=True)
    stock_df.dropna(subset=['close'], inplace=True)

    # normalize weight
    stock_df['prev_w_in_p'] = stock_df['previous_weight'] / \
        stock_df.groupby('date')['previous_weight'].transform('sum')

    stock_df['ini_w_in_p'] = stock_df['initial_weight'] / \
        stock_df.groupby('date')['initial_weight'].transform('sum')

    # calculate weighted pct in portfolio
    stock_df['portfolio_pct'] = stock_df['pct'] * stock_df['prev_w_in_p']

    # calculate weight in sector TODO: remove
    stock_df['prev_w_in_sectore'] = stock_df['previous_weight'] / \
        stock_df.groupby(['date', 'aggregate_sector'])[
        'previous_weight'].transform('sum')
    stock_df['ini_w_in_sector'] = stock_df['initial_weight'] / \
        stock_df.groupby(['date', 'aggregate_sector'])[
        'initial_weight'].transform('sum')
    # weighted pct in sector TODO: remove
    stock_df['sector_pct'] = stock_df['pct'] * stock_df['prev_w_in_sectore']

    # portfolio return
    stock_df['portfolio_return'] = (stock_df.groupby(
        'ticker')['portfolio_pct'].cumprod() + 1) - 1
    # stock_df['cum_p_pct'] = stock_df.groupby(
    #     'ticker')['portfolio_pct'].cumsum()
    # stock_df['portfolio_return'] = np.exp(stock_df['cum_p_pct']) - 1

    # stock return
    stock_df['return'] = (stock_df.groupby('ticker')['pct'].cumprod() + 1) - 1
    # stock_df['cum_pct'] = stock_df.groupby(
    #     'ticker')['pct'].cumsum()
    # stock_df['return'] = np.exp(stock_df['cum_pct']) - 1

    # drop intermediate columns
    stock_df = stock_df.drop(columns=['cum_p_pct'])

    # risk
    stock_df['risk'] = stock_df.groupby('ticker')['pct']\
        .transform(lambda x: x.rolling(len(x), min_periods=1).std() * math.sqrt(252))

    # fill na aggregate_sector
    stock_df['aggregate_sector'].fillna('其他', inplace=True)
    # sector return
    stock_df['sector_return'] = stock_df['ini_w_in_sector'] * \
        stock_df['return']

    return stock_df


# total return by date
def get_portfolio_evaluation(portfolio_stock, benchmark_stock, profile_df):
    # agg by date
    agg_p_stock = portfolio_stock\
        .groupby('date', as_index=False)\
        .agg({'portfolio_return': 'sum', 'portfolio_pct': 'sum'})
    agg_b_stock = benchmark_stock\
        .groupby('date', as_index=False)\
        .agg({'portfolio_return': 'sum', 'portfolio_pct': 'sum'})
    merged_df = pd.merge(agg_p_stock, agg_b_stock, on=[
                         'date'], how='left', suffixes=('_p', '_b'))

    # portfolio mkt cap
    # mkt_adjustment = pd.DataFrame(profile_df.groupby('date')['weight'].sum())
    # mkt_adjustment.rename(columns={'weight': 'mkt_cap'}, inplace=True)
    # merged_df = merged_df.merge(mkt_adjustment, on=['date'], how='outer')

    for i in range(len(merged_df)):
        if pd.isna(merged_df.loc[i, 'mkt_cap']) and i > 0:
            merged_df.loc[i, 'mkt_cap'] = merged_df.loc[i-1,
                                                        'mkt_cap'] * (1 + merged_df.loc[i, 'portfolio_pct_p'])
    # drop where portfolio_return_p is nan
    merged_df.dropna(subset=['portfolio_return_p'], inplace=True)
    # portfolio pnl TODO seem I can just use current wegith to do this
    merged_df['prev_mkt_cap'] = merged_df['mkt_cap'].shift(1)
    merged_df['pnl'] = merged_df['prev_mkt_cap'] * merged_df['portfolio_pct_p']

    # risk std(pct)
    merged_df['risk'] = merged_df['portfolio_pct_p'].rolling(
        len(merged_df), min_periods=1).std() * math.sqrt(252)

    # active return
    merged_df['active_return'] = merged_df['portfolio_pct_p'] - \
        merged_df['portfolio_pct_b']

    # tracking errro std(active return)
    merged_df['tracking_error'] = merged_df['active_return'].rolling(
        len(merged_df), min_periods=1).std() * math.sqrt(252)

    # cum pnl
    merged_df['cum_pnl'] = merged_df['pnl'].cumsum()

    return merged_df


def get_portfolio_sector_evaluation(portfolio_stock, benchmark_df):
    # aggregate on sector and day
    p_sector_df = portfolio_stock.groupby(['date', 'aggregate_sector'], as_index=False)\
        .agg({'prev_w_in_p': 'sum', 'ini_w_in_p': "sum", "current_weight": 'sum',
              "portfolio_pct": "sum", 'sector_return': "sum", 'ini_w_in_sector': 'sum', "portfolio_return": "sum"})
    b_sector_df = benchmark_df.groupby(['date', 'aggregate_sector'], as_index=False)\
        .agg({'prev_w_in_p': 'sum', 'ini_w_in_p': "sum", "current_weight": 'sum',
              "portfolio_pct": "sum", "portfolio_return": "sum", 'sector_return': "sum", 'ini_w_in_sector': 'sum'})

    # merge portfolio and benchmark
    merge_df = p_sector_df.merge(
        b_sector_df, on=['date', 'aggregate_sector'], how='outer', suffixes=('_p', '_b'))

    # to acomendate bhb result
    merge_df.rename(columns={'sector_return_p': 'return_p',
                    'sector_return_b': 'return_b'}, inplace=True)
    # active return
    merge_df['active_return'] = merge_df['portfolio_return_p'] - \
        merge_df['portfolio_return_b']

    # risk
    merge_df['risk'] = merge_df.groupby('aggregate_sector')['portfolio_pct_p']\
        .transform(lambda x: x.rolling(len(x), min_periods=1).std() * math.sqrt(252))

    # tracking error
    merge_df['tracking_error'] = merge_df.groupby('aggregate_sector')['active_return']\
        .transform(lambda x: x.rolling(len(x), min_periods=1).std() * math.sqrt(252))
    return merge_df


# sector_eval_df = get_portfolio_sector_evaluation(portfolio_stock, benchmark_stock)
# sector_eval_df[sector_eval_df.date == datetime(2021, 10,13)].hvplot.bar(x='aggregate_sector', y=['portfolio_pct_p','portfolio_pct_b'], stacked=True, rot=90, title='sector pct')


def merge_on_date(calculated_ps, calculated_bs):
    p_selected = calculated_ps.reset_index(
    )[['ini_w_in_p', 'portfolio_return', 'date', 'ticker', 'display_name', 'return']]
    b_selected = calculated_bs.reset_index(
    )[['ini_w_in_p', 'portfolio_return', 'date', 'ticker', 'return']]
    merged_stock_df = pd.merge(p_selected, b_selected, on=[
                               'date', 'ticker'], how='outer', suffixes=('_p', '_b'))
    return merged_stock_df


# merged_df = merge_on_date(portfolio_stock, benchmark_stock)


def get_bhb_result(merged_stock_df):
    # merged_stock_df['ini_w_in_p_p'].fillna(0, inplace=True)
    # merged_stock_df['ini_w_in_p_b'].fillna(0, inplace=True)
    # merged_stock_df['portfolio_return_b'].fillna(0, inplace=True)
    # merged_stock_df['portfolio_return_p'].fillna(0, inplace=True)
    # allocation
    merged_stock_df['allocation'] = (merged_stock_df['ini_w_in_p_p'] - merged_stock_df['ini_w_in_p_b']) \
        * merged_stock_df['return_b']

    # selection
    merged_stock_df['selection'] = merged_stock_df['ini_w_in_p_b'] * \
        (merged_stock_df['return_p'] -
         merged_stock_df['return_b'])

    # interaction
    merged_stock_df['interaction'] = (merged_stock_df['ini_w_in_p_p'] - merged_stock_df['ini_w_in_p_b']) * \
        (merged_stock_df['return_p'] -
         merged_stock_df['return_b'])

    # excess
    merged_stock_df['excess'] = merged_stock_df['portfolio_return_p'] - \
        merged_stock_df['portfolio_return_b']

    # replace inf with nan
    # merged_stock_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged_stock_df


def calculate_total_attribution_by_sector(calculated_p_stock, calculated_b_stock):
    sector_view_p = calculated_p_stock.groupby(['date', 'aggregate_sector']).aggregate({
        'prev_w_in_p': 'sum', 'sector_pct': 'sum'})
    sector_view_b = calculated_b_stock.groupby(['date', 'aggregate_sector']).aggregate({
        'prev_w_in_p': 'sum', 'sector_pct': 'sum'})

    sector_view_p['weighted_return'] = sector_view_p.prev_w_in_p * \
        sector_view_p.sector_pct
    sector_view_b['weighted_return'] = sector_view_b.prev_w_in_p * \
        sector_view_b.sector_pct

    merged_df = pd.merge(sector_view_p, sector_view_b, left_index=True,
                         right_index=True, how='outer', suffixes=['_b', '_p'])
    merged_df.fillna(0, inplace=True)
    merged_df['active_return'] = merged_df['weighted_return_p'] - \
        merged_df['weighted_return_b']
    merged_df['allocation'] = (
        merged_df.prev_w_in_p_p - merged_df.prev_w_in_p_b) * merged_df.sector_pct_b
    merged_df['selection'] = (
        merged_df.sector_pct_p - merged_df.sector_pct_b) * merged_df.prev_w_in_p_b
    merged_df['interaction'] = (merged_df.sector_pct_p - merged_df.sector_pct_b) * (
        merged_df.prev_w_in_p_p - merged_df.prev_w_in_p_b)
    merged_df['notinal_return'] = merged_df.allocation + \
        merged_df.selection + merged_df.interaction
    return merged_df.reset_index()


def calculate_total_attribution(calculated_p_stock, calculated_b_stock):
    '''
    using pct between two row's data of ticker to calculate the attribute,
    use this method if need to calculate weekly attribut, yearly attribut, etc.
    '''
    merged_df = pd.merge(calculated_b_stock, calculated_p_stock, on=[
                         'date', 'ticker'], how='outer', suffixes=['_b', '_p'])
    df = merged_df[['pct_p', 'pct_b', 'prev_w_in_p_p',
                    'prev_w_in_p_b', 'ticker', 'date']]
    df.fillna(0, inplace=True)
    df['active_return'] = df.pct_p * \
        df.prev_w_in_p_p - df.pct_b * df.prev_w_in_p_b
    # allocation
    df['allocation'] = (df.prev_w_in_p_p - df.prev_w_in_p_b) * df.pct_b
    df['selection'] = (df.pct_p - df.pct_b) * df.prev_w_in_p_b
    df['interaction'] = (df.pct_p - df.pct_b) * \
        (df.prev_w_in_p_p - df.prev_w_in_p_b)
    df['notional_return'] = df.allocation + df.selection + df.interaction

    daily_bnb_result = df.groupby(['date']).aggregate(
        {'allocation': 'sum', 'selection': 'sum', 'interaction': 'sum', 'notional_return': 'sum', 'active_return': 'sum'})
    daily_bnb_result['date'] = daily_bnb_result.index

    return daily_bnb_result.reset_index(drop=True)
    # return df


# def calculate_cum_return_rate(df: pd.DataFrame):
#     '''
#     calcualte return within a window for each entry of ticker
#     inclusive

#     Parameters
#     ----------
#     df : dataframe
#         sorted dafaframe
#     '''

#     # cum return
#     df['cum_return'] = df.groupby('ticker')['pct'].apply(
#         lambda x: (1 + x).cumprod() - 1).reset_index(level=0, drop=True)

#     return df


def _uniformize_time_series(profile_df):
    '''
    a helper function to create analytic_df 

    make each entry in the time series has the same dimension
    by filling none holding stock that was held in previous period has 0 shares and 0 ini_w

    Parameters
    ----------
    profile_df : dataframe
        portfolio profile dataframe or benchmark profile dataframe

    Returns
    -------
    dataframe
        dataframe with uniformized time series
    '''
    # Get unique time periods
    time_periods = profile_df['time'].unique()
    time_periods = sorted(time_periods)

    # Iterate through time periods
    for i in range(len(time_periods) - 1):
        current_period = time_periods[i]
        next_period = time_periods[i + 1]

        current_df = profile_df[profile_df['time'] == current_period]
        next_df = profile_df[profile_df['time'] == next_period]

        tickers_current = current_df['ticker']
        tickers_next = next_df['ticker']

        # row that has ticker not in tickers_next
        missing_tickers = current_df[~tickers_current.isin(
            tickers_next)]

        # not include ticker is ""
        missing_tickers = missing_tickers[missing_tickers.ticker != ""].copy()
        # missing_tickers = missing_tickers[~missing_tickers.ticker.isna()]

        if len(missing_tickers) != 0:
            missing_tickers.time = next_period
            missing_tickers.shares = 0
            missing_tickers.ini_w = 0
            profile_df = pd.concat(
                [profile_df, missing_tickers], ignore_index=True)
    # reset index
    return profile_df.reset_index(drop=True)


def create_analytic_df(price_df, profile_df):
    '''
    create a df for analysis processing

    filling information from profile df to stock price df

    '''
    # daily stock price use begin of the date, need to convert profile_df day to begin of the date
    # profile_df['time'] = profile_df['time'].map(
    #     lambda x: datetime(x.year, x.month, x.day))

    # make every time entry the same dimension
    uni_profile_df = _uniformize_time_series(profile_df)

    # TODO handle rename column here
    df = price_df.merge(uni_profile_df, on=['ticker', 'time'], how='outer')
    df.sort_values(by=['ticker', 'time'], inplace=True)

    # fill close price with ave_price in profile df if exist
    if 'ave_price' in df.columns:
        df.loc[df['ave_price'].notna(
        ), 'close'] = df[df['ave_price'].notna()]['ave_price']

    # add sector, aggregate_sector, display_name and name to missing rows
    grouped = df.groupby('ticker')
    df['sector'] = grouped['sector'].fillna(method='ffill')
    df['aggregate_sector'] = grouped['aggregate_sector'].fillna(method='ffill')
    df['display_name'] = grouped['display_name'].fillna(method='ffill')
    df['name'] = grouped['name'].fillna(method='ffill')

    # assign missing ini_w
    df['ini_w'] = grouped['ini_w'].fillna(method='ffill')

    # assign missing shares, benchmark doesn't have shares
    if ('shares' in df.columns):
        df['shares'] = grouped['shares'].fillna(method='ffill')

        # calcualte handling fee
        df['handling_fee'] = grouped['shares'].diff()
        # TODO: currently not needed so set to 0 for both cases
        df['handling_fee'] = df['handling_fee'].apply(
            lambda x: 0 if x > 0 else 0)
        df['handling_fee'] = df['handling_fee'] * HANDLE_FEE * df['close']

    # fill rest_cap for portfolio
    if ('rest_cap' in df.columns):
        df['rest_cap'] = grouped['rest_cap'].fillna(method='ffill')

    # remove stock price entry where stock has been sold
    df.dropna(subset=['ini_w'], inplace=True)
    if 'rest_cap' in df.columns:
        # handle entry has no stock
        df = df[~((df.close.isna()) & (df.ticker != ""))]
        df = df[(df['ini_w'] != 0) | (df.ticker == "")].copy()
    else:
        df.dropna(subset=['close'], inplace=True)
        df = df[df['ini_w'] != 0].copy()

    # remove where weight is 0

    return df


def calculate_attributes_between_dates(start, end, calculated_p_stock, calculated_b_stock):
    '''
    calculate the attributes to explain the active return between two time series entries, the time series entry
    right after or at start and another time serie right before or at end
    return a df with attributes to explain the active return between start and end time series
    '''
    p_ranged_df = calculated_p_stock[(calculated_p_stock.date >= start) & (
        calculated_p_stock.date <= end)]
    b_ranged_df = calculated_b_stock[(calculated_b_stock.date >= start) & (
        calculated_b_stock.date <= end)]

    p_end_df = p_ranged_df[p_ranged_df.date == p_ranged_df.date.max()]
    p_concat = pd.concat([p_start_df, p_end_df])
    # pct is unweighted return
    p_concat['pct'] = p_concat.groupby('ticker')['close'].pct_change()
    p_concat = p_concat.dropna(subset=['pct'])
    p_concat['prev_w_in_p'] = p_concat['ticker'].map(
        lambda x: p_start_df[p_start_df.ticker == x]['prev_w_in_p'].values[0])
    # p_concatp_concat[['date', 'display_name', 'pct',
    #           'close', 'prev_w_in_p', 'ini_w_in_p']]
    # return and weight of benchmark
    b_start_df = b_ranged_df[b_ranged_df.date == b_ranged_df.date.min()]
    b_end_df = b_ranged_df[b_ranged_df.date == b_ranged_df.date.max()]
    b_concat = pd.concat([b_start_df, b_end_df])
    b_concat['pct'] = b_concat.groupby('ticker')['close'].pct_change()
    b_concat = b_concat.dropna(subset=['pct'])
    b_concat['prev_w_in_p'] = b_concat['ticker'].map(
        lambda x: b_concat[b_concat.ticker == x]['prev_w_in_p'].values[0])
    # b_concat = b_concat[['date', 'display_name', 'pct',
    #           'close', 'prev_w_in_p', 'ini_w_in_p']]
    merged_df = pd.merge(b_concat, p_concat, on=[
                         'ticker', 'date'], suffixes=('_b', '_p'), how='outer')
    df = merged_df[['display_name_p', 'display_name_b', 'ticker',
                    'pct_b', 'pct_p', 'prev_w_in_p_b', 'prev_w_in_p_p']].copy()

    # indicate weather stock is in portfolio
    df['in_portfolio'] = False
    df.loc[df.display_name_p.notnull(), 'in_portfolio'] = True

    # fill display_name
    df['display_name_p'] = df['display_name_p'].fillna(df['display_name_b'])
    df['display_name_b'] = df['display_name_b'].fillna(df['display_name_p'])

    # treat nan weight and pct as 0
    df.fillna(0, inplace=True)

    # allocation, selection, interaction, notional return, active return
    df['allocation'] = (df.prev_w_in_p_p - df.prev_w_in_p_b) * df.pct_b
    df['selection'] = (df.pct_p - df.pct_b) * df.prev_w_in_p_b
    df['interaction'] = (df.pct_p - df.pct_b) * \
        (df.prev_w_in_p_p - df.prev_w_in_p_b)
    df['notional_return'] = df.allocation + df.selection + df.interaction
    # weighted return
    df['return'] = df.prev_w_in_p_p * df.pct_p
    # weight * prev_w is the weighted return
    df['active_return'] = df.prev_w_in_p_p * \
        df.pct_p - df.prev_w_in_p_b * df.pct_b

    return df


def calculate_cum_pnl(df: pd.DataFrame):
    '''return df with cumulative pnl within a window
    Parameters
    ----------
    df : dataframe
       sorted dataframe on time with ticker and pnl collumn
    '''
    grouped = df.groupby('ticker')
    df['cum_pnl'] = grouped['pnl'].cumsum()
    return df


def change_resolution(df, freq='W'):
    '''
    aggregate by keeping the first entry of the freq period,
    the resolution of the df, default to weekly
    '''
    df['freq'] = pd.to_datetime(df['date']).dt.to_period(freq)
    return df.groupby('freq').first().reset_index()


def calculate_pnl(df):
    '''
    patch df with pnl column

    pnl is calculated using pct
    '''
    df.sort_values(by=['time'], inplace=True)
    grouped = df.groupby('ticker')
    prev_cash = grouped.cash.shift(1)
    df['pnl'] = prev_cash * df['pct'] + df['handling_fee']


def calculate_pct(df):
    '''
    calculate pct using close price
    '''
    df.sort_values(by=['time'], inplace=True)
    grouped = df.groupby('ticker')
    df['pct'] = grouped['close'].pct_change()


def calculate_norm_pct(df):
    '''
    use weight to calculate the norm pct
    '''
    df['norm_pct'] = df.weight * df.pct


def calculate_weight_using_cash(df):
    '''
    patch df with current weight for each entry
    use cash to calculate weight

    Parameters
    ----------
    df : dataframe
        dataframe with processed cash column

    '''
    df['weight'] = float('nan')
    grouped = df.groupby('time')
    df.weight = grouped.cash.transform(lambda x: x / x.sum())


def calculate_cash(df):
    '''
    patch df with cash column
    cash = shares * close

    Parameters
    ----------
    df : dataframe
        dataframe with processed shares and close column
    '''
    df['cash'] = df['shares'] * df['close']


def calculate_weight_using_pct(df):
    '''
    calculate weight using weight column

    calculate benchmark stock using this, since benchmark stock
    doesn't have share information

    Parameters
    ----------
    df: dataframe
        dataframe with weight, pct on closing and ini_w columns
    '''
    df.sort_values(by=['time'], inplace=True)
    grouped = df.groupby('ticker')
    for _, group in grouped:
        prev_row = None
        for index, row in group.iterrows():
            if prev_row is None:
                prev_row = df.loc[index]
                continue
            df.loc[index, 'weight'] = prev_row['weight'] * (1 + row['pct'])
            prev_row = df.loc[index]
    # normalize weight
    grouped = df.groupby('time')
    normed_weight = grouped['weight'].transform(lambda x: x / x.sum())
    df['weight'] = normed_weight


def calculate_periodic_BHB(agg_b, agg_p):
    '''
    calculate periodic BHB for each ticker entry

    agg_df's ts entry must have following columns
    1. ticker 
    2. prev_weight: weight at the begin of the period
    3. pct: unweighted return within that period


    Note:
    ----
    if only one entry in a period, the return will be nan,

    Parameters
    ----------
    agg_b : pd.DataFrame
        aggregated benchmark analytic_df
    agg_p : pd.DataFrame
        aggregated portfolio analytic_df

    Returns
    -------
    pd.DataFrame
        periodic BHB result contain allocation, interaction, selection, nominal_active_return and active_return

    '''
    # merge both df on selected column
    agg_b['in_benchmark'] = True
    agg_p['in_portfolio'] = True
    selected_column = ['ticker', 'aggregate_sector',
                       'prev_weight', 'return', 'period', 'display_name', 'pct']
    columns_to_fill = ['pct_b', 'pct_p', 'prev_weight_p', 'prev_weight_b']
    merged_df = pd.merge(agg_b[['in_benchmark'] + selected_column],
                         agg_p,
                         how='outer',
                         on=['period', 'ticker'],
                         suffixes=('_b', '_p'))

    merged_df[columns_to_fill] = merged_df[columns_to_fill].fillna(0)

    # complement fill aggregate_sector and display_name
    post_process_merged_analytic_df(merged_df)

    # calculate active return
    merged_df['return_p'] = merged_df['pct_p'] * \
        merged_df['prev_weight_p']
    merged_df['return_b'] = merged_df['pct_b'] * \
        merged_df['prev_weight_b']
    merged_df['active_return'] = merged_df['return_p'] - \
        merged_df['return_b']

    # allocation, interaction, selection and nominal active return
    merged_df['allocation'] = (
        merged_df.prev_weight_p - merged_df.prev_weight_b) * merged_df.pct_b
    merged_df['interaction'] = (merged_df.pct_p - merged_df.pct_b) \
        * (merged_df.prev_weight_p - merged_df.prev_weight_b)
    merged_df['selection'] = (
        merged_df.pct_p - merged_df.pct_b) * merged_df.prev_weight_b
    merged_df['notional_active_return'] = merged_df['allocation'] + \
        merged_df['interaction'] + merged_df['selection']
    return merged_df


def select_first_last_stock_within_window(df, start, end):
    start = datetime.combine(start, datetime.min.time())
    end = datetime.combine(end, datetime.max.time())
    croped_df = df[df.time.between(start, end, inclusive='both')]
    grouped = croped_df.groupby('ticker')
    first_df = croped_df.loc[grouped.time.idxmin()]
    last_df = croped_df.loc[grouped.time.idxmax()]
    return pd.concat([first_df, last_df])


def post_process_merged_analytic_df(merged_df):
    '''
    fill nan in some column on merged analytic_df

    patch aggregate_sector, display_name, in_portfolio, in_benchmark,

    '''
    # merge both
    merged_df['in_portfolio'].fillna(False, inplace=True)
    merged_df['in_benchmark'].fillna(False, inplace=True)

    # handle aggregated sector view
    if 'aggregate_sector_b' in merged_df.columns:
        # complement fill aggregate_sector and display_name
        merged_df['aggregate_sector_b'].fillna(
            merged_df['aggregate_sector_p'], inplace=True)

    if 'display_name_b' in merged_df.columns:
        merged_df["display_name_b"].fillna(
            merged_df.display_name_p, inplace=True)
        merged_df.rename(columns={'aggregate_sector_b': 'aggregate_sector',
                                  'display_name_b': 'display_name',
                                  }, inplace=True)
        merged_df.drop(columns=['aggregate_sector_p',
                                'display_name_p'], inplace=True)


def aggregate_analytic_df_by_period(df, freq):
    '''
    return an aggregated analytic_df with weekly, monthly, yearly or daily frequency

    each ticker will have 1 rows for each period, 
    cash is the value at the end of the period.
    shares is the # of shares at end of the period.
    prev_weight is the weight of that ticker entry at end of previous period.
    log_return is sum of log_return within the period.
    weight is the weight of that ticker entry at end of the period.
    return is from last of previous period to last of current period.

    Parameters
    ----------
    df : pd.DataFrame
        analytic_df, dateframe of stock price has weight, log_return information
    freq : str
        weekly: 'W-MON' start on tuesday end on monday,
        monthly: 'M', 
        yearly: 'Y', 
        daily: "D"

    Returns
    -------
    pd.DataFrame
        aggregated analytic_df with weekly, monthly, yearly or daily frequency
    '''
    # first ticker at every period start and last ticker at every period end
    df['period'] = df.time.dt.to_period(freq)
    s_df = df.loc[df.groupby(['ticker', 'period']).time.idxmin()]
    l_df = df.loc[df.groupby(['ticker', 'period']).time.idxmax()]

    agg_df = pd.concat([s_df, l_df])

    # unweighted return for each ticker at each period
    agg_df.sort_values('time', inplace=True)
    agg_df['pct'] = agg_df.groupby(['ticker', 'period']).close.pct_change()

    # prev_weight is the weight at began of the week
    agg_df['prev_weight'] = agg_df.groupby(
        ['ticker', 'period']).weight.shift(1)

    agg_df = agg_df.loc[agg_df.groupby(['ticker', 'period']).time.idxmax()]

    return agg_df


def aggregate_bhb_df(df, by="total"):
    '''
    return a aggregate view of attribute result

    Parameters
    ----------
    df : pd.DataFrame
        attribute result dataframe
    by : str, optional
        sector or total, by default "total"
    '''
    keys = ['period', 'aggregate_sector'] if by == 'sector' else ['period']
    agg_df = df.groupby(keys)[['active_return',
                               'allocation',
                               'interaction',
                               'selection',
                               'notional_active_return']].sum()
    return agg_df


def daily_return(df: pd.DataFrame):
    '''
    patch df with daily return
    helper function for get_portfolio_anlaysis
    '''
    prev_ws = df.groupby('ticker')['weight'].shift(1)
    df['return'] = df.pct * prev_ws


def agg_to_daily_sector(df: pd.DataFrame):
    '''
    aggregate a analytic df to daily sector view
    '''
    on_column = {'return': 'sum',
                 'aggregate_sector': 'first', 'weight': 'sum', }
    if 'cash' in df.columns:
        on_column['cash'] = 'sum'
    if 'pnl' in df.columns:
        on_column['pnl'] = 'sum'
    agg_df = df.groupby(['time', 'aggregate_sector']).agg(on_column)
    agg_df = agg_df.reset_index(level=1, drop=True).reset_index()
    agg_df['period'] = agg_df.time.dt.to_period('D')
    # keep the largest time for each day
    agg_df = agg_df[agg_df.groupby(
        'period')['time'].transform(max) == agg_df['time']]
    return agg_df


def agg_to_daily(df: pd.DataFrame):
    '''
    aggreate a analytic df to overal all daily view, if multiple entry in a date only keeping the last ts entry
    '''
    df['period'] = df.time.dt.to_period('D')
    on_column = {'return': 'sum', 'period': 'first'}
    if 'cash' in df.columns:
        on_column['cash'] = 'sum'
    if 'pnl' in df.columns:
        on_column['pnl'] = 'sum'
    if 'rest_cap' in df.columns:
        on_column['rest_cap'] = 'first'

    # aggretate by sum row with the same date
    agg_df = df.groupby('time').agg(on_column).reset_index()

    # keep only row have max time in each day
    agg_df = agg_df[agg_df.groupby(
        'period')['time'].transform(max) == agg_df['time']]
    return agg_df.reset_index()


def calculate_cum_return(df):
    '''
    calculate cumulative return for each ticker entry or whole portfolio
    '''
    if 'ticker' in df.columns:
        df['cum_return'] = df.groupby('ticker')['return'].apply(
            lambda x: (1 + x).cumprod() - 1).reset_index(level=0, drop=True)
    else:
        df['cum_return'] = (df['return'] + 1).cumprod() - 1


def calculate_max_draw_down(df, on):
    """
    find the max draw down on column

    dd: (peak - though) / peak
    mdd: cumulative max dd
    """
    tmp_df = df[['period', on]].copy()

    # Calculate the drawdown for each day
    tmp_df['cum_max'] = tmp_df[on].cummax()
    tmp_df['drawdown'] = (tmp_df['cum_max'] - tmp_df[on]) / tmp_df['cum_max']

    # Find the maximum drawdown and the corresponding date for each day
    max_drawdown = tmp_df['drawdown'].cummax()
    df[f'{on}_max_drawdown'] = max_drawdown
    df[f'{on}_drawdown'] = tmp_df['drawdown']


def get_draw_down(analytic_p, analytic_b):
    '''
    get draw down by pnl and accumulative return

    Parameter
    ---------
    df: pd.DataFrame
        analytic df
    '''
    # aggregate to daily
    agg_p = agg_to_daily(analytic_p)
    agg_b = agg_to_daily(analytic_b)

    # sort by day
    agg_b.sort_values(by=['period'], inplace=True)
    agg_p.sort_values(by=['period'], inplace=True)

    # calculate portfolio total cap
    agg_p['total_cap'] = agg_p['cash'] + agg_p['rest_cap']

    # calculate accumulative return
    agg_p['cum_pnl'] = agg_p['total_cap'].diff()

    # merge
    merged_df = pd.merge(
        agg_p, agg_b, on=['period'], how='outer', suffixes=('_p', '_b'))
    merged_df.sort_values('period', inplace=True)

    # pnl max draw down
    calculate_max_draw_down(merged_df, 'cum_pnl')

    # total capital max draw down
    calculate_max_draw_down(merged_df, 'total_cap')

    return merged_df


def get_portfolio_anlaysis(analytic_p, analytic_b, benchmark_df):
    '''
    return df contain daily pnl, daily return, accumulative return
    risk and tracking error of portfolio and benchmark

    used by the portfolio summary component
    '''
    selected_b_df = benchmark_df[benchmark_df.time.between(
        analytic_p.time.min(), analytic_p.time.max())].copy()

    # calculate return of benchmark
    selected_b_df['pct'] = selected_b_df['close'].pct_change()
    if not selected_b_df.empty:
        selected_b_df['return'] = selected_b_df['close'] / \
            selected_b_df.iloc[0]['close'] - 1
        selected_b_df['return'] = pd.Series(dtype=float)

    # aggregate by exact time
    analytic_p = analytic_p.groupby('time')\
        .agg({'cash': 'sum', 'rest_cap': 'first', 'pnl': 'sum'})\
        .reset_index()
    analytic_b = analytic_b.groupby('time')\
        .agg({'return': 'sum'})\
        .reset_index()

    # merge
    # merged_df = pd.merge(
    #     analytic_p, analytic_b, on=['time'], how='outer', suffixes=('_p', '_b'))
    # TODO: remove duplication, only keep one record for each day
    merged_df = analytic_p.merge(
        selected_b_df, on=['time'],
        how='outer', suffixes=('_p', '_b'))

    merged_df.sort_values('time', inplace=True)

    # portfolio pnl
    merged_df['total_cap'] = merged_df['cash'] + merged_df['rest_cap']
    merged_df['pnl'] = merged_df.total_cap.diff().fillna(0)

    # portfolio return
    # merged_df.iloc[0, merged_df.columns.get_loc('return_p')] = 0
    merged_df['cum_pnl'] = merged_df['pnl'].cumsum()
    if merged_df.empty:
        merged_df['cum_return_p'] = pd.Series(dtype=float)
    else:
        merged_df['cum_return_p'] = merged_df['cum_pnl'] / \
            merged_df.loc[0, 'total_cap']
    merged_df['return_p'] = (merged_df['cum_return_p'] + 1).pct_change()
    merged_df['return_p'].fillna(0, inplace=True)

    # benchmark return
    # merged_df.rename(columns={'return': 'return_b'}, inplace=True)
    # merged_df.iloc[0, merged_df.columns.get_loc('return_b')] = 0
    # merged_df['return_b'].fillna(0, inplace=True)
    # merged_df['cum_return_b'] = (merged_df['return_b'] + 1).cumprod() - 1

    merged_df['pct'].fillna(0, inplace=True)
    merged_df['close'].fillna(method='bfill', inplace=True)
    merged_df['return_b'] = merged_df['close'].pct_change()
    if not merged_df.empty:
        merged_df['cum_return_b'] = merged_df['close'] / \
            merged_df.loc[0, 'close'] - 1
    else:
        merged_df['cum_return_b'] = pd.Series(dtype=float)

    # risk

    merged_df['risk'] = merged_df['return_p'].expanding(
        min_periods=1).std() * math.sqrt(252)
    # active return
    merged_df['active_return'] = merged_df['return_p'] - merged_df['return_b']
    # tracking error
    merged_df['tracking_error'] = merged_df['active_return']\
        .expanding(min_periods=1).std() * math.sqrt(252)

    return merged_df
