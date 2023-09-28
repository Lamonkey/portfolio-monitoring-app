import datetime as dt
from sqlalchemy import create_engine, text
import pandas as pd
from streamz import Stream
import utils as utils
import api
import pytz
import table_schema as ts
import db_operation as db
from log import Log
import processing
from tornado import gen
import os
from tqdm import tqdm
# import settings
# fetch new stock price
stock_price_stream = Stream()
event = Stream(asynchronous=True)
# display notification to 
notification_queue = Stream()
# log
log = Log('instance/log.json')
# save stock price to db
# stock_price_stream.sink(save_stock_price)
# from dask.distributed import Client
# client = Client()
# import nest_asyncio
# nest_asyncio.apply()
# import settings

# run using  --setup
current_path = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_path, "..", 'instance', 'local.db')
db_url = f'sqlite:///{db_dir}'

def get_most_recent_profile(type):
    table_name = 'benchmark_profile' if type == 'benchmark' else 'portfolio_profile'
    query = f"SELECT * FROM {table_name} WHERE date = (SELECT MAX(date) FROM {table_name})"
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(query, con=conn)
        # convert date to datetime object
        df['date'] = pd.to_datetime(df['date'])
        return df


def update_stocks_details_to_db():
    '''override stocks_details table with new stocks detail

    Table Schema
    ------------
    'display_name', 'name', 'start_date', 'end_date', 'type', 'ticker',
       'sector', 'aggregate_sector'
    '''
    df = api.get_all_stocks_detail()
    # validation
    if not _validate_schema(df, ts.STOCKS_DETAILS_TABLE_SCHEMA):
        raise ValueError(
            'df has different schema than STOCKS_DETAILS_TABLE_SCHEMA')
    with create_engine(db_url).connect() as conn:
        df.to_sql(ts.STOCKS_DETAILS_TABLE, con=conn,
                  if_exists='replace', index=False)


def need_to_update_stocks_price(delta_time):
    # convert p_portfolio.date[0] to timezone-aware datetime object
    tz = pytz.timezone('Asia/Shanghai')
    # get stock price df
    with create_engine(db_url).connect() as conn:
        # check if a table exist
        if not conn.dialect.has_table(conn, 'stocks_price'):
            return True
        else:
            query = "SELECT * FROM stocks_price WHERE time = (SELECT MAX(time) FROM stocks_price)"
            most_recent_price = pd.read_sql(query, con=conn)
            most_recent_price.time = pd.to_datetime(most_recent_price.time)
            date_time = tz.localize(most_recent_price.time[0].to_pydatetime())
            if utils.time_in_beijing() - date_time > delta_time:
                return True
            else:
                return False


def add_details_to_stock_df(stock_df):
    with create_engine(db_url).connect() as conn:
        detail_df = pd.read_sql(ts.STOCKS_DETAILS_TABLE, con=conn)
        merged_df = pd.merge(stock_df, detail_df[
            ['sector', 'name',
             'aggregate_sector',
             'display_name',
             'ticker']
        ], on='ticker', how='left')
        merged_df['aggregate_sector'].fillna('其他', inplace=True)
        return merged_df


def _validate_schema(df, schema):
    '''
    validate df has the same columns and data types as schema

    Parameters
    ----------
    df: pd.DataFrame
    schema: dict
        {column_name: data_type}

    Returns
    -------
    bool
        True if df has the same columns and data types as schema
        False otherwise
    '''

    # check if the DataFrame has the same columns as the schema
    if set(df.columns) != set(schema.keys()):
        return False
    # check if the data types of the columns match the schema
    # TODO: ignoring type check for now
    # for col, dtype in schema.items():
    #     if df[col].dtype != dtype:
    #         return False
    return True


def save_stock_price_to_db(df: pd.DataFrame):
    print('saving to stock to db')
    with create_engine(db_url).connect() as conn:
        df.to_sql('stocks_price', con=conn, if_exists='append', index=False)


def update_portfolio_profile_to_db(portfolio_df):
    '''overwrite the portfolio profile table in db, and trigger a left fill on benchmark, stock price
    and recomputation of analysis
    '''

    if (_validate_schema(portfolio_df, ts.PORTFOLIO_TABLE_SCHEMA)):
        raise ValueError(
            'uploaded portfolio_df has different schema than PORTFOLIO_DB_SCHEMA')
    try:
        with create_engine(db_url).connect() as conn:
            portfolio_df[ts.PORTFOLIO_TABLE_SCHEMA.keys()].to_sql(
                ts.PORTFOLIO_TABLE, con=conn, if_exists='replace', index=False)
            
    except Exception as e:
        print(e)
        raise e


def right_fill_stock_price():
    '''
    update all stocks price until today.

    if no benchmark profile, terminate without warning
    default start date is the most recent date in benchmark profile
    '''
    most_recent_benchmark = db.get_most_recent_benchmark_profile()
    most_recent_stocks_price = db.get_most_recent_stocks_price()

    # fetch all stocks price until today
    stocks_dates = most_recent_stocks_price.time
    b_dates = most_recent_benchmark.date
    if len(b_dates) == 0:
        return
    start = stocks_dates[0] if len(stocks_dates) > 0 else b_dates[0]
    end = utils.time_in_beijing()

    # frequency is set to daily
    if end - start > dt.timedelta(days=1):
        new_stocks_price = _fetch_all_stocks_price_between(start, end)
        db.append_to_stocks_price_table(new_stocks_price)


def _fetch_all_stocks_price_between(start, end):
    '''
    patch stock price db with all daily stock price within window
    inclusive on both start and end date

    Parameters
    ----------
    start : datetime 
        start date inclusive
    end: datetime
        end date inclusive

    Returns
    -------
    None
    '''
    # all trading stocks available between start day and end date
    all_stocks = db.get_all_stocks()
    selected_stocks = all_stocks[(all_stocks.start_date <= end) & (
        all_stocks.end_date >= start)]
    tickers = selected_stocks.ticker.to_list()
    # fetch stock price and append to db
    stock_price = api.fetch_stocks_price(
        security=tickers,
        start_date=start,
        end_date=end,
        frequency='daily')
    # drop where closing price is null
    stock_price.dropna(subset=['close'], inplace=True)
    return stock_price


def right_fill_bechmark_profile():
    '''
    right fill the benchmark profile table

    fill any missing entries between the most recent date in benchmark profile and today
    if no benchmark profile, fill from most recent date in portfolio profile to today
    if no portfolio profile, terminate without warning
    '''
    # get most recent date in benchmark profile
    b_ends = db.get_most_recent_benchmark_profile().date
    # get todays date
    today = utils.time_in_beijing()
    # most recent portfolio dates
    p_ends = db.get_most_recent_portfolio_profile().date
    # if portfolio is empty, terminate
    if len(p_ends) == 0:
        return
    # if no benchmark profile, start is the most recent date in benchmark profile, end is today
    elif len(b_ends) == 0:
        start = p_ends[0]
        end = today
    # start is most recent benchmark, end is today
    else:
        start = b_ends[0]
        end = today

    # fetch and update
    new_entry = api.fetch_benchmark_profile(start, end)
    detailed_new_entry = utils.add_details_to_stock_df(new_entry)
    db.append_to_benchmark_profile(detailed_new_entry)


def left_fill_benchmark_profile():
    '''
    left fill the benchmark profile table,

    fill any missing entries between the earliest date in portfolio profile and the earliest date in benchmark profile
    if no portfolio profile, terminate without warning
    if no benchmark profile, the span would be from the earliest date in portfolio profile to the most recent date in portfolio profile
    '''
    # get starttime of benchmark profile
    b_starts = db.get_oldest_benchmark_profile().date

    # get starttime of portfolio profile
    p_starts = db.get_oldest_portfolio_profile().date

    # update window range 
    if len(p_starts) == 0:
        # if no portfolio profile, terminate
        return
    elif len(b_starts) == 0:
        # use start and end date of portfolio profile if no benchmark profile entry
        p_start = p_starts[0]
        b_start = db.get_most_recent_portfolio_profile().date[0]
    else:
        b_start = b_starts[0]
        p_start = p_starts[0]
    if p_start < b_start:
        # back fill benchmark profile
        new_entry = api.fetch_benchmark_profile(p_start, b_start)
        detailed_new_entry = utils.add_details_to_stock_df(new_entry)
    
        # append to db
        db.append_to_benchmark_profile(detailed_new_entry)
        # return detailed_new_entry
    # else do nothing


def left_fill_stocks_price():
    '''
    left fill stock price
    fill missing entries between the oldest date in benchmark 
    profile and the oldest date in stock price table

    if no benchmark profile, terminate without warning
    if no stock price table, the span would be from 
    the oldest date in benchmark profile to the most recent date in benchmark profile


    '''
    # use benchmark because benchmari profile only update once a month
    p_start = db.get_oldest_benchmark_profile().date
    # get oldest time in stock price table
    stock_start = db.get_oldest_stocks_price().time
    # if no portfolio profile, terminate
    if len(p_start) == 0:
        return
    # no stock price, span the entire portfolio profile
    elif len(stock_start) == 0:
        start = p_start[0]
        end = db.get_most_recent_benchmark_profile().date[0]
    else:
        start = p_start[0]
        end = stock_start[0]

    delta_time = dt.timedelta(days=365)
    with tqdm(total=(end - start) / delta_time, colour='green', desc='Fetching stock price') as pbar:
        while start < end:
            # fetch and update
            new_entry = _fetch_all_stocks_price_between(start, start + delta_time)
            db.append_to_stocks_price_table(new_entry)
            # skip previous end date since it is inclusive
            start = min(start + delta_time + dt.timedelta(days=1), end)
            pbar.update(1)


def updaet_benchmark_to_db():
    '''
    update daily benchmark weight
    '''
    pass


def get_stocks_in_profile(profile_df):
    ticker_list = profile_df.ticker.unique().tolist()
    stocks_df = db.get_stocks_price(ticker_list)
    return stocks_df


def batch_processing():
    '''perform when portfolio or benchmark is updated'''
    portfolio_p = db.get_all_portfolio_profile()
    benchmark_p = db.get_all_benchmark_profile()
    p_stocks_df = get_stocks_in_profile(portfolio_p)
    b_stocks_df = get_stocks_in_profile(benchmark_p)

    # temperaraly handle rename date to time
    portfolio_p.rename(
        columns={'date': 'time', 'weight': 'ini_w'}, inplace=True)
    benchmark_p.rename(columns={'date': 'time'}, inplace=True)

    # normalize weight in benchmark
    grouped = benchmark_p.groupby('time')
    benchmark_p['ini_w'] = grouped['weight'].transform(lambda x: x / x.sum())
    with tqdm(total=9, colour='green', desc='Processing') as pbar:
        # add profile information into stock price
        analytic_b = processing.create_analytic_df(b_stocks_df, benchmark_p)
        pbar.update(1)
        analytic_p = processing.create_analytic_df(p_stocks_df, portfolio_p)
        pbar.update(1)
        # p stock weigth
        processing.calculate_cash(analytic_p)
        pbar.update(1)
        processing.calculate_weight_using_cash(analytic_p)
        pbar.update(1)
        processing.calculate_pct(analytic_p)
        pbar.update(1)
        processing.daily_return(analytic_p)
        pbar.update(1)

        # b stock weight
        analytic_b.sort_values(by=['time'], inplace=True)
        grouped = analytic_b.groupby('ticker')
        analytic_b['pct'] = grouped['close'].pct_change()
        processing.calculate_weight_using_pct(analytic_b)
        pbar.update(1)
        processing.daily_return(analytic_b)
        pbar.update(1)

        # pnl
        processing.calculate_pnl(analytic_p)
        pbar.update(1)

    # need to crop on left side of benchmark
    analytic_b = analytic_b[analytic_b['time'] >= analytic_p.time.min()].copy()

    db.save_portfolio_analytic_df(analytic_p)
    db.save_benchmark_analytic_df(analytic_b)


def left_fill():
    left_fill_benchmark_profile()
    left_fill_stocks_price()


def handle_portfolio_update():
    '''
    execute when portfolio is updated, 
    left fill benchmark and stock price

    update method is idempotent, so it is safe to call multiple times
    '''
    left_fill_benchmark_profile()
    print("left fill benchmark profile")
    left_fill_stocks_price()
    print('left fill stock price db')
    batch_processing()
    print('done processing')


async def daily_update():
    '''
    left and right fill stock price and benchmark weight based on portfolio

    the sequence of the update matter,
    specifically the benchmark profile need to be updated first before stock,
    cause update method of stock price depend on the benchmark profile

    '''
    last_update = log.get_time('daily_update')
    # less than today 9am, since it need to force to update at 9
    if last_update is None or utils.time_in_beijing() - last_update >= dt.timedelta(days=1):
        print("running daily update")

        # update benchmark index, this need to be done before update stock price
        left_fill_benchmark_profile()
        right_fill_bechmark_profile()
        print("updated benchmark profile")
        # update stock price
        left_fill_stocks_price()
        right_fill_stock_price()
        print("updated stocks price")
        # update all stock detail
        update_stocks_details_to_db()
        print("updated stocks details")
        log.update_log('daily_update')
    else:
        print("no update needed")
    batch_processing()
    print("updated analytic")


def update():
    '''
    run only once, update stock price and benchmark profile
    '''
    print("Checking stock_price table")
    # collect daily stock price until today in beijing time
    if need_to_update_stocks_price(dt.timedelta(days=1)):
        print("Updating stock_price table")
        # stock_df = update_stock_price()
        stock_df = add_details_to_stock_df(stock_df)
        save_stock_price_to_db(stock_df)
        stock_price_stream.emit(stock_df)




@gen.coroutine
def handle_event(e):
    if e == "update_portfolio":
        print("handling portfolio update")
        handle_portfolio_update()
        print("done handling portfolio update")


event.sink(handle_event)
