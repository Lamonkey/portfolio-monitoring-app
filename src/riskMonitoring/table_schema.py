'''
create df schema for db
'''
PORTFOLIO_TABLE = 'portfolio_profile'
PORTFOLIO_TABLE_SCHEMA = {
    'ticker': str,
    'shares': int,
    'date': 'datetime64[ns]',
    'sector': str,
    'aggregate_sector': str,
    'display_name': str,
    'name': str,
    'cash': float,
    'rest_cap': float,
    'weight': float,
    'ave_price': float
}
STOCKS_DETAILS_TABLE = 'all_stock_info'
STOCKS_DETAILS_TABLE_SCHEMA = {
    'display_name': str,
    'name': str,
    'start_date': 'datetime64[ns]',
    'end_date': 'datetime64[ns]',
    'type': str,
    'ticker': str,
    'sector': str,
    'aggregate_sector': str
}

STOCKS_PRICE_TABLE = 'stocks_price'
STOCKS_PRICE_TABLE_SCHEMA = {
    'time': 'datetime64[ns]',
    'ticker': str,
    'open': float,
    'close': float,
    'high': float,
    'low': float,
    'volume': int,
    'money': float,
}


BENCHMARK_TABLE = 'benchmark_profile'
BENCHMARK_TABLE_SCHEMA = {
    'date': 'datetime64[ns]',
    'weight': float,
    'display_name': str,
    'ticker': str,
    'sector': str,
    'aggregate_sector': str,
    'name': str
}

USER_TABLE = 'user_info'
USER_TABLE_SCHEMA = {
    'username': str,
    'email': str,
    'password': str,
    'role': str
}

BENCHMARK_PRICE_TABLE = 'benchmark_price'
BENCHMARK_PRICE_TABLE_SCHEMA = {
    'open': float,
    'close': float,
    'high': float,
    'low': float,
    'volume': float,
    'money': float,
    'time': 'datetime64[ns]',
 }