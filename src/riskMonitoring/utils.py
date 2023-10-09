import pytz
import datetime
import io
import pandas as pd
from riskMonitoring import db_operation as db
from sqlalchemy import create_engine
import os


def clip_df(start, df: pd.DataFrame, on='time', end=None):
    '''
    return a copy of df between start and end date inclusive
    '''
    # start of that day
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    # end of that day

    if end is None:
        return df[df[on] >= start].copy()
    else:
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
        return df[df[on].between(start, end, inclusive='both')].copy()


def time_in_beijing(strip_time_zone=True):
    '''
    return current time in Beijing as datetime object
    '''
    tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.datetime.now(tz)
    if strip_time_zone:
        dt = dt.replace(tzinfo=None)
    return dt


def add_details_to_stock_df(stock_df):
    '''return df adding sector, aggregate sector, display_name, name to it

    Parameters
    ----------
    stock_df: pd.DataFrame
        the dataframe contain ticker columns

    Returns
    -------
    merged_df: pd.DataFrame
        the dataframe with sector, aggregate sector, display_name and name added

    '''
    # special handling for benchmark profile
    detail_df = db.get_all_stocks_infos()
    if 'display_name' in stock_df.columns:
        stock_df.drop(columns=['display_name'], inplace=True)

    merged_df = pd.merge(stock_df, detail_df[
        ['sector', 'name',
            'aggregate_sector',
            'display_name',
            'ticker']
    ], on='ticker', how='left')

    merged_df['aggregate_sector'].fillna('其他', inplace=True)
    return merged_df


def convert_string_to_datetime(date_string, time_zone="Asia/Shanghai"):
    '''
    Convert a string to a datetime object with the timezone by default,
    Shanghai
    '''
    dt = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f')
    tz = pytz.timezone(time_zone)
    dt = tz.localize(dt)
    return dt


def create_stocks_entry_from_excel(byte_string):
    '''create stock entry from excel file
    Parameters
    ----------
    byte_string: bytes
        the byte string of the excel file
    Returns
    -------
    new_stock_entry: list
        [{ticker:str, shares:int, mean_price: float, date:datetime.datetime}]
        the list of stock entry
    '''
    uploaded_df = None
    with io.BytesIO(byte_string) as f:
        uploaded_df = pd.read_excel(f, index_col=None)

    # throw exception if doesn't have required columns
    if not set(['证券代码', '持仓数量', '平均建仓成本', 'time_stamp', 'cash']).issubset(uploaded_df.columns):
        raise Exception('Missing required columns in excel file')

    # Define the regular expression pattern to match the string endings
    pattern = r'\.(sz|sh)$'

    # Define the replacement strings for each match group
    replacements = {'.sz': '.XSHE', '.sh': '.XSHG'}

    # Use the str.replace method with the pattern and replacements
    uploaded_df['证券代码'] = uploaded_df['证券代码'].str.lower()
    uploaded_df['证券代码'] = uploaded_df['证券代码'].str.replace(
        pattern, lambda m: replacements[m.group()], regex=True)

    new_stock_entry = [
        dict(ticker=ticker, shares=shares, date=time,
             mean_price=mean_price, rest_cap=cash)
        for ticker, shares, mean_price, time, cash in zip(
            uploaded_df['证券代码'],
            uploaded_df['持仓数量'],
            uploaded_df['平均建仓成本'],
            pd.to_datetime(uploaded_df['time_stamp']),
            uploaded_df['cash']
        )]

    return new_stock_entry


def style_number(vals):
    '''color negative number as red, positive as green
    Parameters
    ----------
    vals: df columns
        the columns to be styled
    Returns
    -------
    list
        the list of style

    '''
    return ['color: red' if v < 0 else 'color: green' for v in vals]


def create_share_changes_report(df):
    '''Create a markdown report of the share changes for certain date
    Parameters
    ----------
    df: pd.DataFrame
        the dataframe of profile for a specific date
    Returns
    -------
    markdown: str
    '''

    date_str = df.date.to_list()[0].strftime('%Y-%m-%d %H:%M:%S')
    markdown = f"### {date_str}\n\n"
    markdown += 'Ticker | Display Name | Share Changes\n'
    markdown += '--- | --- | ---\n'
    for _, row in df.iterrows():
        share_changes = row['share_changes']
        # Apply green color to positive numbers and red color to negative numbers
        if share_changes > 0:
            share_changes_str = f'<span style="color:green">{share_changes}</span>'
        elif share_changes < 0:
            share_changes_str = f'<span style="color:red">{share_changes}</span>'
        else:
            share_changes_str = str(share_changes)
        markdown += '{} | {} | {}\n'.format(row['ticker'],
                                            row['display_name'], share_changes_str)
    return markdown


def create_html_report(result: list[tuple]):
    '''
    a flex box with 2 flex item on each row where justified space-between

    Parameters
    ----------
    result: list of tuple
        (title, value, type)
        title: str, title to display
        value: any, value to display
        type: str, used to format value

    Returns
    -------
    html: str
    '''
    style = '''
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
'''

    def _get_color(num):
        return 'green' if num >= 0 else 'red'

    def _format_percentage_number(num):
        return f'{round(num * 100, 2)}%'

    def _create_flex_item(result_entry):
        key, value, value_type = result_entry
        return f"""
            <div>
            <p style="margin: 0;">{key}</p>
            <p style='color: {_get_color(value)}; margin: 0;'>{_format_percentage_number(value)}</p>
            </div>
        """
    html = f"""
        <div class="compact-container">
            {''.join([_create_flex_item(entry) for entry in result])}
        </div>
    """
    return style + html


def update_legend_name(fig, colname_to_name):
    '''
    update plotly legend name using colname_to_name map
    '''
    fig.for_each_trace(lambda t: t.update(name=colname_to_name.get(t.name, t.name),
                                          legendgroup=colname_to_name.get(
        t.name, t.name),

    ))


def validate_stock_json(data):
    '''
    check all stock in stock json match the required format

    Parameters
    ----------
    data: list of stock entry
        [{ticker, shares, ave_prie, rest_cap, time}]
    '''
    # all has date
    all_has_date = all([entry.get('date', None) is not None for entry in data])
    if not all_has_date:
        raise Exception('All stock entry should have date')
    # all has ticker
    all_has_ticker = all(
        [entry.get('ticker', None) is not None for entry in data])
    if not all_has_ticker:
        raise Exception('All stock entry should have ticker')

    # all has aver_price
    all_has_ave_price = all(
        [entry.get('ave_price', None) is not None for entry in data])
    if not all_has_ave_price:
        raise Exception('All stock entry should have ave_price')

    # all has rest_cap
    all_has_rest_cap = all(
        [entry.get('rest_cap', None) is not None for entry in data])
    if not all_has_rest_cap:
        raise Exception('All stock entry should have rest_cap')

    # check if shares is int
    all_shares_int = all([isinstance(entry['shares'], int) for entry in data])
    if not all_shares_int:
        raise Exception('All stock entry should have integer shares')

    # check if ave_price is float or int
    all_ave_price_float = all(
        [isinstance(entry['ave_price'], (float, int)) for entry in data])
    if not all_ave_price_float:
        raise Exception(
            'All stock entry should have float or integer ave_price')

    # check if reset cash is int or float
    all_rest_cap_float = all(
        [isinstance(entry['rest_cap'], (float, int)) for entry in data])
    if not all_rest_cap_float:
        raise Exception(
            'All stock entry should have float or integer rest_cap')

    # check if date string match y-m-d h:m:s
    try:
        [datetime.datetime.strptime(
            entry['date'], '%Y-%m-%d %H:%M:%S') for entry in data]
    except ValueError:
        raise Exception(
            'All stock entry should have date string match %Y-%m-%d %H:%M:%S')

    # all entry should have the same date and same cash
    all_same_date = all([entry['date'] == data[0]['date'] for entry in data])
    if not all_same_date:
        raise Exception('All stock entry should have the same date')
    all_same_cash = all([entry['rest_cap'] == data[0]['rest_cap']
                        for entry in data])
    if not all_same_cash:
        raise Exception('All stock entry should have the same rest_cap')

    # all entry have different ticker
    all_different_ticker = len(
        set([entry['ticker'] for entry in data])) == len(data)
    if not all_different_ticker:
        raise Exception('All stock entry should have different ticker')

    # all entry have positive shares
    all_positive_shares = all([entry['shares'] > 0 for entry in data])
    if not all_positive_shares:
        raise Exception('All stock entry should have positive shares')

    # all entry have positive ave_price
    all_positive_ave_price = all([entry['ave_price'] > 0 for entry in data])
    if not all_positive_ave_price:
        raise Exception('All stock entry should have positive ave_price')

    # all entry have positive rest_cap
    all_positive_rest_cap = all([entry['rest_cap'] > 0 for entry in data])
    if not all_positive_rest_cap:
        raise Exception('All stock entry should have positive rest_cap')

    return True


def create_profile_from_json(json_data, use_ave_price=True):
    '''
    create a profile df from json data

    Parameters
    ----------
    json_data: list of stock entry
        [{ticker, shares, ave_prie, rest_cap, time}]
    use_ave_price: bool
        if True, use ave_price to calculate weight, otherwise use latest close price
    '''

    # convert date to datetime object
    df = pd.DataFrame(json_data)

    # convert date string to datetime object
    df['date'] = df.apply(lambda x: datetime.datetime.strptime(
        x['date'], '%Y-%m-%d %H:%M:%S'), axis=1)
    df['date'] = pd.to_datetime(df['date'])
    all_stock_info = db.get_all_stocks_infos()

    # convert ticker to jq format 
    df['ticker'] = df.ticker.str.lower()

    # check if all ends in sz or sh
    if not df['ticker'].str.endswith(('.sz', '.sh', '.xshg', '.xshe')).all():
        raise Exception('All ticker should end with .SZ, .SH or .XSHG, .XSHE ')
    
    # replace suffix to jq format
    df['ticker'] = df.ticker.str.replace(
        r'\.(sz|sh)$', lambda m: '.XSHE' if m.group() == '.sz' else '.XSHG', regex=True)

    df = pd.merge(df, all_stock_info, on='ticker', how='left')

    # check if any display name is null
    if df['display_name'].isnull().any():
        null_ticker = df[df['display_name'].isnull()]['ticker'].tolist()
        raise Exception(f'{null_ticker} is not a valid ticker')

    # calculate weight using ave_price
    df['cash'] = df['ave_price'] * df['shares']
    df['weight'] = df['cash'] / df['cash'].sum()
    # drop start_date, end_date, type 
    df.drop(columns=['start_date', 'end_date', 'type'], inplace=True)
    return df
