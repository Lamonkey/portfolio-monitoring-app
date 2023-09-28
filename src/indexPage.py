import panel as pn
from sidebar import SideNavBar
import indexPageComponents
import processing
import db_operation as db
import api
import utils as utils
import pandas as pd

pn.extension('mathjax')
pn.extension('plotly')
pn.extension('tabulator')
pn.extension(notifications=True)

template = pn.template.ReactTemplate(
    title='Portfolio一览',
    # side_bar_width=200,
    collapsed_sidebar=True,
    sidebar=[SideNavBar()],
    cols={'lg': 12, 'md': 8, 'sm': 3, 'xs': 3, 'xxs': 3},
    save_layout=True,
    prevent_collision=False,


)

analytic_p = db.get_portfolio_analytic_df()
analytic_b = db.get_benchmark_analytic_df()

if len(analytic_p) == 0:
    template.main[0, 6:12] = pn.pane.HTML(
        '<h1>没有检测到portfolio,请先上传portfolio</h1>')


else:
    # a large number to let flex-basis takes full width
    max_width = 4000
    min_width = 300
    styles = {'border': '1px solid black', 'padding': '10px'}
    stock_overview = indexPageComponents.BestAndWorstStocks(
        analytic_df=analytic_p,
        styles=styles,
        title="股票表现排名"
    )
    composation_card = indexPageComponents.PortfolioCompositionCard(
        analytic_df=analytic_p,
        max_width=max_width,
        min_width=min_width,
        styles=styles

    )
    return_analysis = indexPageComponents.ReturnAnlaysisCard(
        calculated_p_stock=analytic_p,
        calculated_b_stock=analytic_b,
        title='回报率分析',
        max_width=max_width,
        min_width=min_width,
        styles=styles
    )
    total_return_card = indexPageComponents.OverviewCard(
        b_stock_df=analytic_b,
        p_stock_df=analytic_p,
        styles=styles,
        value=(0, 20))
    drawdown_card = indexPageComponents.DrawDownCard(
        calculated_p_stock=analytic_p,
        max_width=max_width,
        min_width=min_width,
        styles=styles,
    )

    sector_performance = indexPageComponents.SectorPerformance(
        analytic_p=analytic_p,
        analytic_b=analytic_b,
        title='行业表现',
        styles=styles
    )
    # template.main.extend([drawdown_card, stock_overview, composation_card, monthly_return_card, total_return_card])
    # template.main[0, :] = top_header
    foo_df = processing.get_draw_down(analytic_p)
    # top header
    cum_pnl_trend = indexPageComponents.TrendPlot(
        title='累计pnl', data_series=foo_df.cum_pnl)
    cum_return_trend = indexPageComponents.TrendPlot(
        title='累计回报率', data_series=foo_df.cum_return)
    cum_return_dd = indexPageComponents.TrendPlot(
        title='最大回撤', data_series=foo_df.cum_return_dd)

    template.main[0, 0:4] = cum_pnl_trend
    template.main[0, 4:8] = cum_return_trend
    template.main[0, 8:12] = cum_return_dd
    # first column
    template.main[1:6, 0:4] = total_return_card
    template.main[6:12, 0:4] = return_analysis
    # second column
    template.main[1:8, 4:8] = composation_card
    template.main[8:12, 4:8] = sector_performance
    # template.main[8:12, 4:8] = drawdown_card

    # third column
    template.main[1:5, 8:12] = stock_overview
    template.main[5:8, 8:12] = drawdown_card

    # put a spacer at last row cross all columns
    template.main[-1, 0:12] = pn.Spacer(height=100)

    # calculate max drawdown here

    latest_analytic_df = analytic_p[analytic_p.time ==
                                    analytic_p.time.max()].copy()
    latest_portfolio_df = foo_df[foo_df.period == foo_df.period.max()].copy()

    def stream_data():
        # use most recent row in the analytic db
        global latest_analytic_df, latest_portfolio_df

        last_m_df = api.fetch_stocks_price(
            security=latest_analytic_df.ticker.tolist(),
            end_date=utils.time_in_beijing(),
            frequency='1m',
            count=1,
        )
        concat_df = pd.concat([latest_analytic_df, last_m_df], axis=0)
        concat_df.sort_values(by=['time'], inplace=True)
        concat_df.reset_index(drop=True, inplace=True)

        # give a warning if there is duplicate ticker and date and do nothing
        if concat_df.duplicated(subset=['ticker', 'time']).any():
            pn.state.notifications.warning(
                'Stream 还在运行，但是没有新数据\n' +
                f'最新数据 {concat_df.time.max().strftime("%Y-%m-%d %H:%M:%S")}',
                duration=0)
            return

        # calcualte pct
        concat_df['pct'] = concat_df.groupby('ticker')['close'].pct_change()

        # calculate weight using pct
        concat_df['weight'] = concat_df.groupby('ticker')\
            .apply(lambda x: (x['pct'] + 1) * x['weight'].shift(1))\
            .reset_index(drop=True, level=0)
        # norm weight
        concat_df['weight'] = concat_df\
            .groupby('time')['weight']\
            .apply(lambda x: x/x.sum())\
            .reset_index(drop=True, level=0)

        # calculate return
        concat_df['return'] = concat_df['pct'] * concat_df['weight']

        # calculate pnl using return and previous cash
        concat_df['pnl'] = concat_df.groupby('ticker')\
            .apply(lambda x: x['return'] * x['cash'].shift(1))\
            .reset_index(drop=True, level=0)

        # calculate cumulative max drawdown

        # update latest row
        latest_analytic_df = concat_df[concat_df.time ==
                                       concat_df.time.max()].copy()

        # calculate cumulative return, cumulative pnl, max drawdown on cumulative return
        portfolio_df = processing.agg_to_daily(latest_analytic_df)
        portfolio_return = portfolio_df['return'].values[0]
        portfolio_pnl = portfolio_df['pnl'].values[0]
        latest_portfolio_df['cum_return'] = latest_portfolio_df['cum_return'] * \
            (1 + portfolio_return)
        latest_portfolio_df['cum_pnl'] = latest_portfolio_df['cum_pnl'] + portfolio_pnl
        latest_portfolio_df['ex_max_cum_return'] = max(
            latest_portfolio_df['cum_return'].values[0], latest_portfolio_df['ex_max_cum_return'].values[0])
        latest_portfolio_df['cum_return_dd'] = latest_portfolio_df['cum_return'] / \
            latest_portfolio_df['ex_max_cum_return'] - 1

        # stream cumulative pnl
        cum_pnl_trend.stream_data(
            entries=latest_portfolio_df.cum_pnl.to_list())

        # stream cumulative return
        cum_return_trend.stream_data(
            entries=latest_portfolio_df.cum_return.to_list())

        # stream max drawdown
        cum_return_dd.stream_data(
            entries=latest_portfolio_df.cum_return_dd.to_list())

    def toggle_stream_btn():
        callback = pn.state.add_periodic_callback(
            stream_data, period=60*1000, start=False)

        def toggle_streaming_callback(event):
            if callback.running or callback._updating:
                # If the callback is running, stop it
                callback.stop()
                stream_btn.name = '开启stream'
                stream_btn.value = True
                stream_btn.button_type = 'success'
                print('Stopped streaming')
            else:
                # If the callback is not running, start it
                callback.start()
                stream_btn.value = False
                stream_btn.name = '关闭stream'
                stream_btn.button_type = 'danger'
                print('Started streaming')
        btn = pn.widgets.Button(
            name='开启stream', button_type='success', size_policy='stretch_width')
        btn.on_click(toggle_streaming_callback)
        return btn

    stream_btn = toggle_stream_btn()
    template.sidebar.append(stream_btn)

template.servable()
