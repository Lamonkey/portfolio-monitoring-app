import pytz
import datetime
import io
import pandas as pd
from riskMonitoring import db_operation as db
from sqlalchemy import create_engine
import os



def clip_df(start, end, df: pd.DataFrame, on='time'):
    '''
    return a copy of df between start and end date inclusive
    '''
    return df[df.time.between(start, end, inclusive='both')].copy()


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
    fig.for_each_trace(lambda t: t.update(name=colname_to_name.get(t.name, t.name),
                                          legendgroup=colname_to_name.get(
        t.name, t.name),

    ))
