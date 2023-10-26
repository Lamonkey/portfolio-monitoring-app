import panel as pn
from panel.viewable import Viewer
from typing import Dict

class UserCredentialManger(Viewer):
    pn.extension('tabulator')

    def __init__(self, credentials: Dict[str, str], **params):
        self.credentialsTable = pn.widgets.Tabulator()
        self.discardBtn = pn.widgets.Button(name="Discard")
        self.applyBtn = pn.widgets.Button(name="Apply")
        self.


