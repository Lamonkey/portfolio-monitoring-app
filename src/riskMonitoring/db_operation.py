'''
Abstraction for saving data to db
'''
import pandas as pd
import riskMonitoring.table_schema as ts
from sqlalchemy import create_engine
import os

current_path = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_path, "../..", 'instance', 'local.db')
db_url = f'sqlite:///{db_dir}'


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


def get_most_recent_profile(type):
    table_name = 'benchmark_profile' if type == 'benchmark' else 'portfolio_profile'
    query = f"SELECT * FROM {table_name} WHERE date = (SELECT MAX(date) FROM {table_name})"
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(query, con=conn)
        # convert date to datetime object
        df['date'] = pd.to_datetime(df['date'])
        return df


def get_all_benchmark_profile():
    '''return all entries in the benchmark profile table'''
    return _get_all_row(ts.BENCHMARK_TABLE)


def append_to_benchmark_profile(df):
    '''append new entry to benchmark profile table'''

    # handle possible duplication caused by right fill
    most_recent_dates = get_most_recent_benchmark_profile().date
    if len(most_recent_dates) > 0:
        date = most_recent_dates[0]
        # drop df.date == date
        df = df[df.date != date]
        if len(df) != 0:
            _append_df_to_db(df, ts.BENCHMARK_TABLE, ts.BENCHMARK_TABLE_SCHEMA)
    else:
        _append_df_to_db(df, ts.BENCHMARK_TABLE, ts.BENCHMARK_TABLE_SCHEMA)


def get_most_recent_benchmark_profile():
    '''return the most recent entry in the benchmark profile table'''
    return _get_most_recent(ts.BENCHMARK_TABLE)


def get_most_recent_profile(type):
    table_name = 'benchmark_profile' if type == 'benchmark' else 'portfolio_profile'
    query = f"SELECT * FROM {table_name} WHERE date = (SELECT MAX(date) FROM {table_name})"
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(query, con=conn)
        # convert date to datetime object
        df['date'] = pd.to_datetime(df['date'])
        return df


def _get_oldest(table_name, ts_column='date'):
    query = f"SELECT * FROM {table_name} WHERE {ts_column} = (SELECT MIN({ts_column}) FROM {table_name})"
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(query, con=conn)
        df[ts_column] = pd.to_datetime(df[ts_column])
        return df


def get_oldest_stocks_price():
    df = _get_oldest(ts.STOCKS_PRICE_TABLE, ts_column='time')
    return df


def get_oldest_portfolio_profile():
    df = _get_oldest(ts.PORTFOLIO_TABLE)
    return df


def get_oldest_stocks_proce():
    df = _get_oldest(ts.STOCKS_PRICE_TABLE, ts_column='time')
    return df


def get_oldest_benchmark_profile():
    df = _get_oldest(ts.BENCHMARK_TABLE)
    return df


def _get_most_recent(table_name, ts_column='date'):
    '''return the most recent entry in the table'''
    query = f"SELECT * FROM {table_name} WHERE {ts_column} = (SELECT MAX({ts_column}) FROM {table_name})"
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(query, con=conn)
        # convert date to datetime object
        df[ts_column] = pd.to_datetime(df[ts_column])
        return df


def get_all_portfolio_profile():
    df = _get_all_row(ts.PORTFOLIO_TABLE)
    # df['date'] = pd.to_datetime(df['date'])
    return df


def get_most_recent_portfolio_profile():
    df = _get_most_recent(ts.PORTFOLIO_TABLE)
    return df


def get_most_recent_stocks_price():
    df = _get_most_recent(ts.STOCKS_PRICE_TABLE, ts_column='time')
    return df


def _append_df_to_db(df, table_name, schema):
    # validation
    if not _validate_schema(df, schema):
        raise Exception(
            f'VALIDATION_ERROR: df does not have the same schema as the table {table_name}')
    with create_engine(db_url).connect() as conn:
        df.to_sql(table_name, con=conn, if_exists='append', index=False)


def append_to_stocks_price_table(df):
    '''append new entry to stocks price table'''
    _append_df_to_db(df, ts.STOCKS_PRICE_TABLE, ts.STOCKS_PRICE_TABLE_SCHEMA)


def get_all_stocks_infos():
    '''
    get all stocks information

    Returns
    -------
    pd.DataFrame
        all stocks information
    '''
    with create_engine(db_url).connect() as conn:
        all_stocks = pd.read_sql(ts.STOCKS_DETAILS_TABLE, con=conn)
        return all_stocks


def replace_stock_detail_with(df):
    if not _validate_schema(df, ts.STOCKS_DETAILS_TABLE_SCHEMA):
        raise Exception(
            f'VALIDATION_ERROR: df does not have the same schema as the table {ts.STOCKS_DETAILS_TABLE}')
    with create_engine(db_url).connect() as conn:
        df.to_sql(ts.STOCKS_DETAILS_TABLE, con=conn,
                  if_exists='replace', index=False)


def save_portfolio_analytic_df(df):
    table_name = 'analytic_p'
    with create_engine(db_url).connect() as conn:
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)


def get_portfolio_analytic_df():
    table_name = 'analytic_p'
    with create_engine(db_url).connect() as conn:
        # if table doesn't exist meanning don't have portfolio yet
        try:
            df = pd.read_sql(table_name, con=conn)
            return df
        except Exception:
            # empty df to accomendate the pipeline
            return pd.DataFrame()


def save_benchmark_analytic_df(df):
    table_name = 'analytic_b'
    with create_engine(db_url).connect() as conn:
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)


def get_benchmark_analytic_df():
    table_name = 'analytic_b'
    with create_engine(db_url).connect() as conn:
        try:
            # if doesn't exist
            df = pd.read_sql(table_name, con=conn)
            return df
        except Exception:
            # return empty df to accomendate the pipeline
            return pd.DataFrame()


def save_analytic_df(df):
    table_name = 'analytic'
    with create_engine(db_url).connect() as conn:
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)


def get_analytic_df():
    table_name = 'analytic'
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(table_name, con=conn)
        return df


def _get_all_row(table_name, ts_column='date'):
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(table_name, con=conn)
        df[ts_column] = pd.to_datetime(df[ts_column])
        return df


def get_all_stocks_price():
    '''
    return all entries in stocks price table
    '''
    return _get_all_row(ts.STOCKS_PRICE_TABLE)


def get_stocks_price(tickers: list[str]):
    '''
    return df of stock price within ticker in stocks price table
    '''
    if len(tickers) == 0:
        # select 0 zero but return df has the same schema
        query = f"SELECT * FROM {ts.STOCKS_PRICE_TABLE} WHERE 1=0"
    elif len(tickers) == 1:
        query = f"SELECT * FROM {ts.STOCKS_PRICE_TABLE} WHERE ticker = '{tickers[0]}'"
    else:
        query = f"SELECT * FROM {ts.STOCKS_PRICE_TABLE} WHERE ticker IN {tuple(tickers)}"
    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(query, con=conn)
        df.time = pd.to_datetime(df.time)

    # drop duplicates
    return df.drop_duplicates(subset=['ticker', 'time'])


def upload_stock_price_to_db(df: pd.DataFrame):
    with create_engine(db_url).connect() as conn:
        df.to_sql(ts.STOCKS_PRICE_TABLE, con=conn,
                  if_exists='append', index=False)


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
