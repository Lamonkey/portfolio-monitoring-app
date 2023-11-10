import numpy as np
import panel as pn
from riskMonitoring.components import sidebar
from typing import Dict
from riskMonitoring.utils import (update_credential_file)
import pandas as pd
import os
import inspect
import importlib
import riskMonitoring.utils as utils

import param
from panel.viewable import Viewer

from riskMonitoring.db_operation import (
    update_user_info,
    get_all_user_info
)
xs = np.linspace(0, np.pi)


class MonitorManger(Viewer):

    def __init__(self, **params):
        # load current config
        self.config = utils.load_monitor_config_json()
        self.load_monitor_module()
        self.multi_choice = pn.widgets.MultiChoice(
            name='Active Alert Module', value=[], options=[], sizing_mode='stretch_width')
        self.save_btn = pn.widgets.Button(name="Save")
        self.test_btn = pn.widgets.Button(name="Test alert")
        self.save_btn.on_click(self.handle_save)
        self.test_btn.on_click(self.handle_test)
        self.testing_email = []
        self.param_column = pn.Column(sizing_mode='stretch_width', scroll=True)
        self.update_muti_select()
        super().__init__(**params)

    def load_monitor_module(self):
        '''
        load monitor module from riskMonitoring.alertMonitor
        '''
        self.module_name = "riskMonitoring.alertMonitor"
        self.module = importlib.import_module(self.module_name)
        self.monitor_and_value = {name: cls for name,
                                  cls in inspect.getmembers(self.module, inspect.isclass)}


    def update_muti_select(self):
        '''
        update the multi_choice 
        '''
        self.multi_choice.options = list(self.monitor_and_value.keys())
        self.multi_choice.value = list([config['name'] for config in self.config])

    @param.depends('multi_choice.value', watch=True)
    def sync(self):
        '''
        sync between the multi_choice and param_column
        '''
        for monitor in self.config:
            name = list(monitor.keys())[0]
            value = list(monitor.values())[0]
            # text_filed
            tf = pn.widgets.TextInput(name=name, value=str(
                value), placeholder=f'param for {name}', sizing_mode='stretch_width')
            self.param_column.append(tf)
            self.multi_choice.value.append(name)

        
    def handle_save(self, _):
        selected_monitors = self.multi_choice.value
        monitor_config = [
            {'name': name, 'param': 1 } for name in selected_monitors
        ]
        utils.save_monitor_config_json(monitor_config)

    def handle_test(self, _):
        print("testing alert...")
        pass

    def __panel__(self):
        self._layout = pn.Column(
            self.multi_choice,
            pn.Row(
                self.save_btn,
                self.test_btn,
                sizing_mode='stretch_width',
            ),
            self.param_column,
            sizing_mode='stretch_width',
        )
        return self._layout


def createUserCredentialManager(**params):
    user_table = get_all_user_info()
    user_tabulator = pn.widgets.Tabulator(
        value=user_table,
        buttons={'Delete': "<i class='fa fa-trash'></i>"},
        layout='fit_data_stretch',
        sizing_mode='stretch_width',
        hidden_columns=['index'],
    )

    add_user_btn = pn.widgets.Button(name="Add User")
    discard_btn = pn.widgets.Button(name="Discard")
    apply_btn = pn.widgets.Button(name="Apply")

    def handle_add_new(e):
        user_tabulator.stream(
            {'username': '', 'password': '', 'email': '', 'role': '1'},
            follow=False)

    def handle_discard(e):
        user_tabulator.value = get_all_user_info()

    def handle_apply(e):
        try:
            update_user_info(user_tabulator.value)
            update_credential_file()
        except Exception as e:
            print(e)
        user_tabulator.value = get_all_user_info()

    def handle_delete_row(e):
        if e.column == 'Delete':
            # copy of user_tabulator.value without e.row
            removed_row_df = user_tabulator.value.drop(e.row)
            user_tabulator.value = removed_row_df.reset_index(drop=True)
        else:
            pass
    # click delete row
    user_tabulator.on_click(handle_delete_row)
    discard_btn.on_click(handle_discard)
    apply_btn.on_click(handle_apply)
    add_user_btn.on_click(handle_add_new)
    return pn.Column(
        user_tabulator,
        pn.Row(add_user_btn,
               discard_btn,
               apply_btn,
               sizing_mode='stretch_width',
               styles={"justify-content": "end"}
               ),
        sizing_mode='stretch_width',
    )


# Instantiate the template with widgets displayed in the sidebar
template = pn.template.ReactTemplate(
    title='ReactTemplate',
    sidebar=sidebar.Component(),
    save_layout=False,
    collapsed_sidebar=True,
)
# Populate the main area with plots, to demonstrate the grid-like API
template.main[:2, :4] = createUserCredentialManager()
template.main[:4, 4:7] = MonitorManger()

# template.main[:3, 6:] = pn.Card(dfi_cosine.hvplot(**plot_opts).output(), title='Cosine')
template.servable()
