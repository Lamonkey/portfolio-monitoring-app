import hvplot.pandas
import numpy as np
import panel as pn
import pandas as pd
from riskMonitoring.components import sidebar
from typing import Dict
from riskMonitoring.utils import (update_credential_file)
import pandas as pd
from riskMonitoring.db_operation import (
    update_user_info,
    get_all_user_info
)
xs = np.linspace(0, np.pi)


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
            {'username': '', 'password': '', 'email': '', 'role':'1'}, 
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

# template.main[:3, 6:] = pn.Card(dfi_cosine.hvplot(**plot_opts).output(), title='Cosine')
template.servable()
