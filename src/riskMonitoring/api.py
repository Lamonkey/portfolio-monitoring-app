
'''
contain method for api call to jqdatasdk
'''
from dotenv import load_dotenv
from datetime import datetime, timedelta
import jqdatasdk as jq
import pandas as pd
from typing import List, Optional
from sqlalchemy import create_engine
import riskMonitoring.table_schema as ts
import os
from tqdm import tqdm
import riskMonitoring.db_operation as db

load_dotenv()
user_name = os.getenv('JQDATA_USER')
password = os.getenv('JQDATA_PASSWORD')


def auth_api(func):
    """
    decorator for function require jqdatasdk api
    """
    def wrapper(*args, **kwargs):

        if (not jq.is_auth()):
            jq.auth(user_name, password)

        result = func(*args, **kwargs)
        return result

    return wrapper


@auth_api
def get_quota():
    return jq.get_query_count()


def aggregate_sector(input: str) -> Optional[str]:
    '''
    mapping from sector to aggregated sector retur None if not found
    this handling is for spotting undefined sector in current mapping 
    later 

    Return: str -- aggregated sector
            None if no mapping
    '''
    mapping = {
        '电气设备I': '工业',
        '建筑装饰I': '工业',
        '交通运输I': '工业',
        '机械设备I': '工业',
        '国防军工I': '工业',
        '综合I': '工业',
        '电子I': '信息与通信',
        '计算机I': '信息与通信',
        '通信I': '信息与通信',
        '传媒I': '信息与通信',
        '纺织服装I': '消费',
        '家用电器I': '消费',
        '汽车I': '消费',
        '休闲服务I': '消费',
        '商业贸易I': '消费',
        '食品饮料I': '消费',
        '美容护理I': '消费',
        '农林牧渔I': '消费',
        '钢铁I': '原料与能源',
        '建筑材料I': '原料与能源',
        '有色金属I': '原料与能源',
        '化工I': '原料与能源',
        '轻工制造I': '原料与能源',
        '煤炭I': '原料与能源',
        '石油石化I': '原料与能源',
        '采掘I': '原料与能源',
        '医药生物I': '医药卫生',
        '公用事业I': '公用事业',
        '环保I': '公用事业',
        '房地产I': '金融与地产',
        '银行I': '金融与地产',
        '非银金融I': '金融与地产'
    }
    # return the first mapping found
    sectors = input.split(" ")
    maped_name = "其他"
    for sector in sectors:
        maped_name = mapping.get(sector, None)
        if maped_name is not None:
            return maped_name

    return maped_name


@auth_api
def get_all_stock_info() -> tuple[pd.DataFrame, List[str]]:
    '''
    return all stock information

    Return
    ------
    tuple: tuple(pd.DataFrame, List[str])
        DataFrame -- display_name | name | start_date | end_date | type
    '''
    error = []
    try:
        df = jq.get_all_securities()
        df['ticker'] = df.index
        df.reset_index(drop=True, inplace=True)
        # df.reset_index(inplace=True)
        return df, error
    except Exception as e:
        error.append(f'get_all_stock_info\n{e}')
        return None, error


@auth_api
def add_detail_to_stocks(df: pd.DataFrame) -> List[str]:
    """
    add display_name, name, sector, and aggregate sector to each stock if not exist already
    return a list of error message

    Args: pd.DataFrame
    ticker | date | weight | sector | aggregate_sector | display_name | name

    Returns: List[str], error messages
    """
    error = []
    df[['sector', 'aggregate_sector']] = df.groupby(
        'ticker')[['sector', 'aggregate_sector']].ffill()
    df[['display_name', 'name']] = df.groupby(
        'ticker')[['display_name', 'name']].ffill()
    not_have_sector = list(
        df[df['aggregate_sector'].isnull()]['ticker'].unique())
    not_have_name = list(df[df['name'].isnull()]['ticker'].unique())
    # sector and aggregate sector
    if len(not_have_sector) != 0:
        try:
            sectors = jq.get_industry(security=not_have_sector)
            df['sector'] = df.apply(lambda x: x.sector if not pd.isna(x.sector)
                                    else " ".join(value['industry_name']
                                                  for value in sectors[x.ticker].values()), axis=1)
            df['aggregate_sector'] = df.apply(
                lambda x: x.aggregate_sector if not pd.isna(x.aggregate_sector)
                else aggregate_sector(x.sector), axis=1
            )
        except Exception as e:
            error.append(f'Error on creaet_sector_information\n{ticker}\n{e}')

    # display_name and name
    if len(not_have_name) != 0:
        try:
            for ticker in not_have_name:
                detail = jq.get_security_info(ticker)
                df.loc[df.ticker.isin(not_have_name)
                       ]['display_name'] = detail.display_name
                df.loc[df.ticker.isin(not_have_name)]['name'] = detail.name
        except Exception as e:
            error.append(f'Error on get display_name and name\n{ticker}\n{e}')

    return error


@auth_api
def update_portfolio_profile(stocks: List[dict], current_p: pd.DataFrame = None) -> tuple[pd.DataFrame, List[str]]:
    """create or update a portfolio profile,
    return a time series of profile

    Parameters
    ----------
    stocks : List[{ticker: Str, shares: float, date:datetime}]

        update profile with a list of stock information

    current_p : pd.DataFrame, optional
        current portfolio profile, default is None

    Returns
    -------
    updated_profile : pd.DataFrame
        ticker | date | weight | sector | aggregate_sector | display_name | name

    error : List[str]
        a list of error message
    """

    error = []
    profile_df = pd.DataFrame(stocks)
    profile_df['sector'] = None
    profile_df['aggregate_sector'] = None

    # add display_name
    try:
        info_df = db.get_all_stocks_infos()
        profile_df = pd.merge(
            profile_df, info_df[['display_name', 'ticker', 'name', 'aggregate_sector', ]], on='ticker', how='left')
    except Exception as e:
        error.append(f'create_portfolio \n{e}')

    # get sector information
    incoming_error = add_detail_to_stocks(profile_df)
    error.extend(incoming_error)

    # concate to existing profile if exist
    if current_p is not None:
        profile_df = pd.concat([profile_df, current_p], ignore_index=True)
        profile_df.drop_duplicates(
            subset=['ticker', 'date'], keep='last', inplace=True)
        profile_df.reset_index(drop=True, inplace=True)

    return profile_df, error


@auth_api
def get_all_stocks_detail():
    '''get df contain all stock display_name, name, sector, aggregate_sector'''
    detail_df = jq.get_all_securities()
    detail_df['ticker'] = detail_df.index
    detail_df.reset_index(drop=True, inplace=True)
    industry_info = jq.get_industry(detail_df.ticker.to_list())
    detail_df['sector'] = detail_df.apply(lambda x: " ".join(
        value['industry_name']for value in industry_info[x.ticker].values()), axis=1)
    detail_df['aggregate_sector'] = detail_df.apply(
        lambda x: aggregate_sector(x.sector), axis=1)
    return detail_df


@auth_api
def get_api_usage():
    return jq.get_query_count()


# @auth_api
# def fetch_stocks_price(profile: pd.DataFrame, start_date: datetime, end_date: datetime, frequency='daily') -> tuple[pd.DataFrame, List[str]]:
#     """
#     Return a dataframe contain stock price between period of time for price in a portfolio profile

#     Arguments:
#         profile {pd.DataFrame} -- ticker | date | weight | sector | aggregate_sector | display_name | name
#         start_date {datetime} -- start date of the period include start date
#         end_date {datetime} -- end date of the period include end date
#         frequency {str} -- resolution of the price, default is daily

#     Returns: Tuple(pd.DataFrame, List[str])
#         pd.DataFrame -- ticker date open close high low volumn money
#         error_message {list} -- a list of error message
#     """
#     error_message = []
#     start_str = start_date.strftime('%Y-%m-%d')
#     end_str = end_date.strftime('%Y-%m-%d')
#     if profile.date.min() < start_date:
#         # hanlde benchmark doesn't have weight on the exact date
#         start_str = profile.date.min().strftime('%Y-%m-%d')

#     ticker = profile['ticker'].to_list()
#     try:

#         data = jq.get_price(ticker, start_date=start_str,
#                             end_date=end_str, frequency=frequency)
#         data.rename(columns={'time': 'date', 'code': "ticker"}, inplace=True)
#         return data, error_message
#     except Exception as e:
#         error_message.append(f'Error when fetching {ticker} \n {e}')
#         return None, error_message


@auth_api
def fetch_stocks_price(**params):
    '''request list of stock price from start_date to end_date with frequency or count'''
    stocks_df = jq.get_price(**params)
    stocks_df.rename(columns={'code': 'ticker'}, inplace=True)

    if params.get('frequency') == 'daily' or params.get('frequency') == '1d':
        # replace time to market close time
        stocks_df['time'] = stocks_df['time'].apply(lambda x: x.replace(hour=15, minute=0, second=0))
    return stocks_df
# jq.get_price(security='600673.XSHG', end_date=datetime.now(), frequency='1m', count=1)


@auth_api
def fetch_benchmark_profile(start_date: datetime, end_date: datetime, delta_time=timedelta(days=30), benchmark="000905.XSHG"):
    '''
    fetch benchmark profile from start_date to end_date with delta_time

    Parameters
    ----------
    start_date : datetime
        start date of the period include start date
    end_date : datetime
        end date of the period include end date
    delta_time : timedelta, optional
        the default is 30 days since the jq api only update index weight once every month
    '''
    if end_date < start_date:
        raise Exception('end_date must be greater than start_date')

    results = []
    with tqdm(total=(end_date - start_date) / delta_time, colour='green', desc='Fetching benchmark') as pbar:
        while start_date < end_date:
            try:
                date_str = start_date.strftime('%Y-%m-%d')
                result = jq.get_index_weights(benchmark, date=date_str)
                results.append(result)
            except Exception as e:
                print(f'Error when fetching {benchmark}\n\
                                    update on {date_str} is missing\n\
                                    {e}')
            start_date += delta_time
            print(1)
            pbar.update(1)
    update_df = pd.concat(results)
    update_df['ticker'] = update_df.index
    update_df['date'] = pd.to_datetime(update_df['date'])
    # update_df.rename({'date': 'time'}, inplace=True, axis=1)
    # remove duplicate row
    update_df = update_df.drop_duplicates(
        subset=['ticker', 'date'], keep='last')
    update_df.reset_index(drop=True, inplace=True)

    # replace time to same date 3pm
    update_df['date'] = update_df['date'].apply(
        lambda x: x.replace(hour=15, minute=0, second=0))
    
    return update_df

# print(fetch_stocks_price(security=['601077.XSHG','300009.XSHE'],end_date=datetime(2023, 9, 26 ,10, 17), frequency='1m',count=1))
