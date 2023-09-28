import panel as pn
import pandas as pd
import numpy as np
from streamz import Stream
from pipeline import stock_price_stream
stream = Stream()
stock_price_stream
pn.extension('tabulator')
pn.extension('vega')

stream_df = pd.DataFrame(columns=['time', 'ticker', 'open', 'close', 'high', 'low',
                                  'volume', 'money', 'in_portfolio', 'in_benchmark', 'aggregate_sector', 'display_name'])

stream_table = pn.widgets.Tabulator(
    stream_df, layout='fit_columns', width=1200, height=1200)
# stream_table


def stream_data(stream_df):
    print('updating stream!!!')
    # stream_df = pd.DataFrame(np.random.randn(5, 5), columns=list('ABCDE'))
    stream_table.stream(stream_df, follow=True)


def create_new_stream():
    stream_df = pd.DataFrame(np.random.randn(5, 5), columns=list('ABCDE'))
    stock_price_stream.emit(stream_df)
# pn.state.add_periodic_callback(create_new_stream, period=1000, count=100)


stock_price_stream.sink(stream_data)
template = pn.template.FastListTemplate(
    title='api monitor')
# stock_price_stream.sink(print)
template.main.extend(
    [stream_table]
)
# )
# stock_price_stream.sink(print)
template.servable()
