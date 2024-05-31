import panel as pn
from riskMonitoring.components import portfolioComposition, sidebar
import riskMonitoring.processing as processing
import riskMonitoring.db_operation as db
import riskMonitoring.api as api
import riskMonitoring.utils as utils
import pandas as pd
from riskMonitoring.components import (
    maxDrawDown,
    trendPlot,
    sectorPerformance,
    overview,
    returnAnalysis,
    bestAndWorstStocks
)

pn.extension('mathjax')
pn.extension('plotly')
pn.extension('tabulator')
pn.extension(notifications=True)

template = pn.template.ReactTemplate(
    title='Portfolio一览',
    # side_bar_width=200,
    collapsed_sidebar=True,
    sidebar=[sidebar.Component()],
    cols={'lg': 12, 'md': 8, 'sm': 3, 'xs': 3, 'xxs': 3},
    save_layout=True,
    prevent_collision=False,


)

analytic_p = db.get_portfolio_analytic_df()
analytic_b = db.get_benchmark_analytic_df()
benchmark_price = db.get_benchmark_price_between(
    analytic_b.time.min(),
    analytic_b.time.max())

if len(analytic_p) == 0:
    template.main[0, 6:12] = pn.pane.HTML(
        '<h1>没有检测到portfolio,请先上传portfolio</h1>')


else:
    # a large number to let flex-basis takes full width
    max_width = 4000
    min_width = 300
    styles = {'border': '1px solid black', 'padding': '10px'}
    stock_overview = bestAndWorstStocks.Component(

        analytic_df=analytic_p,
        styles=styles,
        title="股票表现排名"
    )
    composation_card = portfolioComposition.Component(
        analytic_df=analytic_p,
        max_width=max_width,
        min_width=min_width,
        styles=styles

    )
    return_analysis = returnAnalysis.Component(
        calculated_p_stock=analytic_p,
        calculated_b_stock=analytic_b,
        title='回报率分析',
        max_width=max_width,
        min_width=min_width,
        styles=styles
    )
    total_return_card = overview.Component(
        benchmark_price=benchmark_price,
        b_stock_df=analytic_b,
        p_stock_df=analytic_p,
        styles=styles,
    )
    drawdown_card = maxDrawDown.Component(
        calculated_p_stock=analytic_p,
        analytic_b=analytic_b,
        max_width=max_width,
        min_width=min_width,
        styles=styles,
    )

    sector_performance = sectorPerformance.Component(
        analytic_p=analytic_p,
        analytic_b=analytic_b,
        title='行业表现',
        styles=styles
    )
   
    # TODO: make first row began at 0
    template.main[1:6, 0:4] = total_return_card
    template.main[6:12, 0:4] = return_analysis
    # second column
    template.main[1:8, 4:8] = composation_card
    template.main[8:14, 4:8] = sector_performance
    # template.main[8:12, 4:8] = drawdown_card

    # third column
    template.main[1:5, 8:12] = stock_overview
    template.main[5:8, 8:12] = drawdown_card

    # put a spacer at last row cross all columns
    template.main[15, 0:12] = pn.layout.HSpacer(height=100)


template.servable()
