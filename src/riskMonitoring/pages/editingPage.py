

import panel as pn
from riskMonitoring.utils import create_stocks_entry_from_excel, style_number, create_share_changes_report, time_in_beijing
from bokeh.models.widgets.tables import NumberEditor, SelectEditor
from tranquilizer import tranquilize
import pandas as pd
import riskMonitoring.api as api
import riskMonitoring.db_operation as db
import riskMonitoring.pipeline as pipeline
from riskMonitoring.components.sidebar import Component
import json
from riskMonitoring import utils

pn.extension()
pn.extension('tabulator')
pn.extension('plotly')
pn.extension('floatpanel')
pn.extension(notifications=True)


# the width of iphone se
MIN_COMPONENT_WIDTH = 375
MAX_COMPONENT_WIDTH = 600

@tranquilize(method="POST")
def upload_to_db(data):
    # check if valid json
    try:
        data = json.loads(data)
    except Exception:
        return {'status': 401, 'message': 'not a valid json format'}
    
    # check json match the format
    try:
        utils.validate_stock_json(data)
    except Exception as e:
        return {'status': 402, 'message': str(e)}
    # check if profile df is valid
    try:
        data = utils.create_profile_from_json(data)
    except Exception as e:
        return {'status': 403, 'message': str(e)}
    # check if db already have ticker with same time stamp
    try:
        db.add_new_portfolio(data)
    except Exception as e:
        return {'status': 404, 'message': str(e)}
    # save to db
    # data = json.loads(data)
    # print(data)
    return f'transaction succedd'


def create_portfolio_stream_entry(stocks, portfolio_df):
    '''
    create a df to be stream to portfolio tabulator
    '''
    # create entry with ticker, date and details
    stream_entry = pd.DataFrame(stocks)

    # if have duplicate ticker raise error
    if stream_entry.ticker.duplicated().any():
        raise Exception('VALIDATION_ERROR: plase remove duplicate ticker')

    # raise error if portfolio already have same ticker with same date
    date = stream_entry.date[0]
    selected_df = portfolio_df[portfolio_df.date == date]
    tickers = stream_entry.ticker.tolist()
    filter_out_ticker = selected_df[selected_df.ticker.isin(tickers)].ticker

    if len(filter_out_ticker) > 0:
        raise Exception(
            f'VALIDATION_ERROR: {" ".join(filter_out_ticker)}在{date}已存在,请先删除再添加')

    stream_entry = pipeline.add_details_to_stock_df(stream_entry)

    # calculate share changes, use tmp_df to save intermediate result
    tmp_df = pd.concat([stream_entry, portfolio_df],
                       ignore_index=True, join='outer')
    tmp_df.sort_values(by='date', inplace=True)
    tmp_df['share_changes'] = tmp_df.groupby('ticker').shares.diff()

    # for ticker previous not existing use shares as share_changes
    tmp_df.share_changes = tmp_df.share_changes.fillna(tmp_df.shares)

    # add share_chagnes back to stream_entry
    stream_entry = stream_entry.merge(
        tmp_df[['ticker', 'date', 'share_changes', 'change_saved']],
        on=['ticker', 'date'],
        how='left'
    )

    # indicate not saved
    stream_entry['change_saved'] = False

    # indicate sync to db
    stream_entry['sync_to_db'] = True

    # fill empty ave_price with latest closing price
    # TODO: for now all ave_price is fetching from api
    ticker = stream_entry.ticker.tolist()
    close_price = api.fetch_stocks_price(
        security=ticker, end_date=date, frequency='minute', count=1)[['ticker', 'close']]
    close_price.rename(columns={'close': 'ave_price'}, inplace=True)
    stream_entry = stream_entry.merge(close_price, on='ticker', how='left')

    # calculate cash(mkt_value) and weight
    stream_entry['cash'] = stream_entry.shares * stream_entry.ave_price
    stream_entry['weight'] = stream_entry.cash / stream_entry.cash.sum()
    return stream_entry


def notify(func):
    def wrapper(*args, **kwargs):
        try:
            notifications = func(*args, **kwargs)

            if notifications is not None:
                for notification in notifications:
                    duration = notification.get('duration', 4000)
                    if notification['type'] == 'success':
                        pn.state.notifications.success(
                            notification['description'], duration=duration)
                    elif notification['type'] == 'error':
                        pn.state.notifications.error(
                            notification['description'], duration=duration)
                    elif notification['type'] == 'warning':
                        pn.state.notifications.warning(
                            notification['description'], duration=duration)
                    elif notification['type'] == 'info':
                        pn.state.notifications.info(
                            notification['description'], duration=duration)
                    else:
                        raise Exception('unknow notification type')
        except Exception as e:
            pn.state.notifications.error(str(e), duration=0)
    return wrapper


def app():

    # load portfolio df
    p_profile = db.get_all_portfolio_profile()
    p_profile.sort_values(by=['date'], inplace=True)
    # change in shares for same ticker
    p_profile['share_changes'] = p_profile.groupby(['ticker'])[
        'shares'].diff()
    p_profile['share_changes'] = p_profile['share_changes'].fillna(
        p_profile['shares'])
    # indicate if change is saved
    p_profile['change_saved'] = True
    p_profile['sync_to_db'] = True

    # get all stocks ticker for auto fill
    stock_details = db.get_all_stocks_infos()
    all_tickers = stock_details.ticker.to_list()

    # get most recent portfolio for auto generate entry
    most_recent_portfolio = None
    if len(p_profile) == 0:
        most_recent_portfolio = p_profile
    else:
        most_recent_portfolio = p_profile[p_profile.date == max(
            p_profile.date)]

    # create portfolio table tabulator
    hidden_column = ['index', 'sector', 'name']
    col_to_titles = {'ticker': '证劵代码', 'weight': '权重',
                     'date': '时间', 'aggregate_sector': '分类',
                     'display_name': '名称',
                     'shares': '持仓', 'change_saved': '已同步',
                     'sync_to_db': '存入', 'share_changes': '持仓变化',
                     'cash': '现金', 'ave_price': '平均成本',
                     'rest_cap': '剩余资产',
                     }
    # styling
    tabulator_formatters = {
        # 'float': {'type': 'progress', 'max': 10},
        'sync_to_db': {'type': 'tickCross'},
        'change_saved': {'type': 'tickCross'},
    }
    bokeh_editors = {
        'ticker': SelectEditor(options=all_tickers),
        'shares': NumberEditor(),
    }
    # frozen_columns = ['date','ticker','display_name','shares','sync_to_db','change_saved']

    portfolio_tabulator = pn.widgets.Tabulator(p_profile,
                                               layout='fit_data_stretch',
                                               sizing_mode='stretch_both',
                                               groupby=['date'],
                                               hidden_columns=hidden_column, titles=col_to_titles,
                                               formatters=tabulator_formatters,
                                               editors=bokeh_editors,
                                               pagination='local',
                                               page_size=25,
                                               #    frozen_columns=frozen_columns
                                               )

    portfolio_tabulator.style.apply(style_number, subset=['share_changes'])

    # history tabulator
    history_dt = p_profile[['date', 'sync_to_db', 'change_saved']].copy()
    history_dt = history_dt.groupby('date').agg({
        "sync_to_db": lambda x: all(x),
        'change_saved': lambda x: all(x),
    })
    history_dt['date'] = history_dt.index
    history_dt.reset_index(drop=True, inplace=True)
    history_tabulator = pn.widgets.Tabulator(history_dt,
                                             layout='fit_data_stretch',
                                             formatters=tabulator_formatters,
                                             buttons={'detail': "<i>📋</i>"},
                                             hidden_columns=hidden_column,
                                             sizing_mode='stretch_both',
                                             titles=col_to_titles)
    # perform calculation btn
    force_recalculate_btn = pn.widgets.Button(
        name='重新计算', button_type='primary', sizing_mode='stretch_width')
    # create component
    new_stock_btn = pn.widgets.Button(
        name='增加新股票', button_type='primary', sizing_mode='stretch_width')
    preview_btn = pn.widgets.Button(
        name='预览', button_type='primary', sizing_mode='stretch_width')
    file_input = pn.widgets.FileInput(
        accept='.xlsx', sizing_mode='stretch_width')
    # strip timezone info
    datetime_picker = pn.widgets.DatetimePicker(name='Datetime Picker',
                                                value=time_in_beijing().replace(tzinfo=None),
                                                sizing_mode='stretch_width')

    total_cap_input = pn.widgets.FloatInput(name='剩余资产',
                                            value=0.0,
                                            step=1,
                                            start=0,
                                            sizing_mode='stretch_width')
    upload_to_db_btn = pn.widgets.Button(
        name='保存到数据库', button_type='warning', sizing_mode='stretch_width')
    # emtpy stock_column to display new entires
    stock_column = pn.Column(sizing_mode='stretch_both')
    # floating window row
    floating_windows = pn.Row()

    @notify
    def _update_history_tabulator(action, df=None):
        '''handle update history tabulator'''
        # handle add new entires to view
        if action == 'append' and df is not None:
            index = history_tabulator.value[history_tabulator.value.date ==
                                            df.date[0]].index.to_list()
            if len(index) == 0:
                # drop duplicate date in df
                df = df.drop_duplicates(subset='date', keep='first')
                # if not in history tabulator add new entry
                selected_df = df[['date', 'sync_to_db', 'change_saved']]
                # if stream to empty tabulator, index will be mismatched
                if (len(history_tabulator.value) == 0):
                    history_tabulator.value = selected_df
                else:
                    history_tabulator.stream(
                        df[['date', 'sync_to_db', 'change_saved']], follow=True)
            else:
                # if in history tabulator patch change_saved to false
                history_tabulator.patch({
                    'change_saved': [(index[0], False)]
                }, as_index=True)

            yield {'type': 'warning', 'description': '添加成功请保存'}
        # hanlde editing portoflio tabulator
        elif action == 'edit':
            # mark synced_to_db to false when entry is edited
            date = df
            index = history_tabulator.value[history_tabulator.value.date == date].index.to_list(
            )
            # check if all change saved
            all_saved = all(
                portfolio_tabulator.value[portfolio_tabulator.value.date == date]['change_saved'])
            history_tabulator.patch({
                'change_saved': [(index[0], all_saved)]
            }, as_index=True)
            yield {'type': 'warning', 'description': '修改成功请保存'}
        # handle sync to db
        elif action == 'sync':
            # patch all synced_to_db to true
            indices = history_tabulator.value[
                ~history_tabulator.value['change_saved']].index.to_list()

            # add an offset to address the issue when df is empty index start from 1

            history_tabulator.patch({
                'change_saved': [(index, True) for index in indices]
            }, as_index=True)
            yield {'type': 'success', 'description': '同步成功以更新'}

    @notify
    def delete_stock(row):
        '''delete a stock entry'''
        stock_column.remove(row)
        yield {'type': 'success', 'description': '删除成功', "duration": 1000}

    def create_new_stock_entry(ticker=None, shares=0, ave_price=0.0, disable_ticker=True):
        '''create a new new stock entry component'''

        # remove shares
        delete_btn = pn.widgets.Button(
            name='❌', width=50, height=60, sizing_mode='fixed')

        # ticker input
        ticker_selector = pn.widgets.AutocompleteInput(
            value=ticker,
            name='证劵代码',
            sizing_mode='stretch_width',
            options=all_tickers,
            placeholder='input ticker',

        )
        # num of shares
        share_input = pn.widgets.IntInput(
            name='持仓',
            value=shares,
            step=1,
            start=0,
            sizing_mode='stretch_width')

        row = pn.Row(
            delete_btn,
            ticker_selector,
            share_input,
            width_policy='max',
        )
        delete_btn.on_click(lambda _, row=row: delete_stock(row))
        return row

    def update_stock_column(xlsx_file=None):
        stock_entries = []
        if xlsx_file is not None:
            stocks_list = create_stocks_entry_from_excel(xlsx_file)
            for entry in stocks_list:
                stock_entries.append(create_new_stock_entry(
                    ave_price=entry['mean_price'],
                    ticker=entry['ticker'],
                    shares=entry['shares']))
            # modify time
            datetime_picker.value = stocks_list[0]['date']

            # modify the capital
            total_cap_input.value = stocks_list[0]['rest_cap']

        # update
        stock_column.clear()
        stock_column.extend(stock_entries)

    @notify
    def update_profile_tabulator(e):
        '''add all stocks entry to ui'''
        # TODO: make this idempotent
        new_entries = [dict(ticker=row[1].value,
                            shares=row[2].value,
                            rest_cap=total_cap_input.value,
                            date=datetime_picker.value) for row in stock_column]

        try:
            new_profile = create_portfolio_stream_entry(
                new_entries, portfolio_tabulator.value)
            # update history tabulator
            _update_history_tabulator('append', new_profile)
            _stream_to_portfolio_tabulator(new_profile)
            yield {'type': 'success', 'description': f'已添加{len(new_entries)}条新股票,请保存'}
        except Exception as e:
            raise Exception(e)

    def add_new_stock(e):
        row = create_new_stock_entry()
        stock_column.append(row)

    @notify
    def _stream_to_portfolio_tabulator(entry):
        # not using stream because it will cause index mismatch
        if len(portfolio_tabulator.value) == 0:
            portfolio_tabulator.value = entry

        else:
            portfolio_tabulator.stream(entry, follow=True)
            yield {'type': 'success', 'description': f'添加{len(entry)}条股票'}

    def handle_click_on_history_tabulator(e):
        '''handle click click on history tabulator'''
        if e.column == 'detail':
            row_index = e.row
            date = history_tabulator.value.iloc[row_index]['date']
            date_str = date.strftime("%Y-%m-%d : %H:%M:%S")
            record_df = portfolio_tabulator.value[portfolio_tabulator.value.date == date]
            floatpanel = pn.layout.FloatPanel(create_share_changes_report(
                record_df), name=date_str, margin=20, position='right-top')
            floating_windows.append(floatpanel)

    @notify
    def handle_sync_to_db(e):
        # TODO: change to use profile df instead, because tabulator might not contain all entry (currently have no problem)
        '''sync selected entry to db'''
        new_portfolio = portfolio_tabulator.value

        # only update selected row to db
        selected_portfolio = new_portfolio[new_portfolio['sync_to_db']]

        try:
            db.update_portfolio_profile_to_db(selected_portfolio)
        except Exception as e:
            raise Exception(f'同步到数据库失败,错误信息:{e}')
        # update history tabulator and portfolio tabulator

        # mark changes as saved
        indices = selected_portfolio[~selected_portfolio['change_saved']].index.to_list(
        )
        portfolio_tabulator.patch({
            'change_saved': [(index, True) for index in indices]
        }, as_index=True)
        _update_history_tabulator('sync')
        yield {'type': 'success', 'description': '保存成功'}

    def handle_edit_portfolio_tabulator(e):
        date = portfolio_tabulator.value.iloc[e.row]['date']
        _update_history_tabulator(df=date, action='edit')

    def hanlde_edit_history_tabulator(e):
        # toggle sync on all entry on a date
        if e.column == 'sync_to_db':
            date = history_tabulator.value.iloc[e.row]['date']
            # index of all entry on portfolio tabulator
            indices = portfolio_tabulator.value[portfolio_tabulator.value.date.between(
                date.replace(microsecond=0), date.replace(microsecond=999999))].index.to_list()
            # patch all indices on sync_to_db to e.value
            portfolio_tabulator.patch({
                'sync_to_db': [(index, e.value) for index in indices]
            }, as_index=True)




    @notify
    def handle_force_recalculation(e):
        try:
            yield {'type': 'info', 'description': "开始重新计算可能会花费1分钟以上", 'duration': 0}
            # fill missing benchmark profile
            yield {'type': 'info', 'description': "正在获取benchmark数据", 'duration': 0}
            pipeline.left_fill_benchmark_profile()
            pipeline.right_fill_bechmark_profile()
            # fill missing stock price
            yield {'type': 'info', 'description': "正在更新股票数据", 'duration': 0}
            pipeline.left_fill_stocks_price()
            pipeline.right_fill_stock_price()

            # recalculate
            yield {'type': 'info', 'description': "正在重新计算权重", 'duration': 0}
            pipeline.batch_processing()

            yield {'type': 'info', 'description': '完成✅', 'duration': 0}

        except Exception as e:
            raise Exception(f'重新计算失败,错误信息:{e}')

    # register event handler
    upload_to_db_btn.on_click(handle_sync_to_db)
    preview_btn.on_click(update_profile_tabulator)
    new_stock_btn.on_click(add_new_stock)
    history_tabulator.on_click(
        handle_click_on_history_tabulator
    )
    force_recalculate_btn.on_click(handle_force_recalculation)
    portfolio_tabulator.on_edit(handle_edit_portfolio_tabulator)
    history_tabulator.on_edit(hanlde_edit_history_tabulator)
    # create handler component to add to panel so can be listened to
    upload_xlsx_handler = pn.bind(update_stock_column, file_input)

    # layout

    editor_widget = pn.Column(datetime_picker,
                              upload_to_db_btn,
                              new_stock_btn,
                              preview_btn,
                              force_recalculate_btn,
                              file_input,
                              total_cap_input,
                              pn.widgets.TooltipIcon(
                                  value="用于更新修改持仓信息,默认股票为最近持仓，默认时间为目前北京时间，点击增加新股票按钮,输入股票代码和持仓选择日期(北京时间),点击预览,确认无误后点击保存到数据库。或者直接拖拽excel文件到下方上传按钮"),
                              stock_column,
                              upload_xlsx_handler,
                              scroll=True,
                              sizing_mode='stretch_both',
                              #   styles={'background-color': 'red'}
                              )
    # tooltip
    toolTip2 = pn.widgets.TooltipIcon(
        value="持仓总结,每一行的已同步到数据库代表所做更改是否已同步到数据库,点击保存到数据库将上传所有更改。点击右侧📋按钮查看详细持仓变化报告")

    return editor_widget, pn.Column(history_tabulator, sizing_mode="stretch_both"), pn.Column(portfolio_tabulator, floating_windows, sizing_mode="stretch_both")


# app
editor_widget, history_tabulator, portfolio_tabulator = app()
template = pn.template.ReactTemplate(
    title='portfolio编辑',
    cols={'lg': 12, 'md': 8, 'sm': 3, 'xs': 3, 'xxs': 3},
    collapsed_sidebar=True,
    sidebar=[Component()],
    save_layout=True,
)
template.main[0:3, 0:3] = editor_widget
template.main[3:4, 0:3] = history_tabulator
template.main[0:4, 3:12] = portfolio_tabulator
template.servable()
