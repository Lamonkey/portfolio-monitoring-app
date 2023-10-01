from sqlalchemy import create_engine
import pandas as pd
import riskMonitoring.table_schema as ts
import os

current_path = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_path, "../..", 'instance', 'local.db')
db_url = f'sqlite:///{db_dir}'


def _create_table_with_schema(table_name: str, table_schema: dict):
    with create_engine(db_url).connect() as conn:
        df = pd.DataFrame(
            columns=table_schema.keys()).astype(table_schema)
        df.to_sql(
            table_name, conn, if_exists='replace', index=False)
        return True


def initialize_db():
    # initialize portfolio profile table
    if not _create_table_with_schema(ts.PORTFOLIO_TABLE, ts.PORTFOLIO_TABLE_SCHEMA):
        raise Exception(
            f'INITIALIZATION ERROR: cannot create table {ts.PORTFOLIO_TABLE} ')
    # initialize stocks details table
    if not _create_table_with_schema(ts.STOCKS_DETAILS_TABLE, ts.STOCKS_DETAILS_TABLE_SCHEMA):
        raise Exception(
            f'INITIALIZATION ERROR: cannot create table {ts.STOCKS_DETAILS_TABLE} ')
    # initialize stocks price table
    if not _create_table_with_schema(ts.STOCKS_PRICE_TABLE, ts.STOCKS_PRICE_TABLE_SCHEMA):
        raise Exception(
            f'INITIALIZATION ERROR: cannot create table {ts.STOCKS_PRICE_TABLE} ')
    # initialize benchmark profile table
    if not _create_table_with_schema(ts.BENCHMARK_TABLE, ts.BENCHMARK_TABLE_SCHEMA):
        raise Exception(
            f'INITIALIZATION ERROR: cannot create table {ts.BENCHMARK_TABLE} ')


# allow to be run as script
if __name__ == '__main__':
    initialize_db()
