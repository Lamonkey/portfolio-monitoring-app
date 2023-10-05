from panel.viewable import Viewer
import panel as pn


class Component(Viewer):

    def __init__(self, **params):
        self.pages = {
            '编辑Portfolio': "/editingPage",
            '主页': "/indexPage",
        }
        self.styles = {
            'text-decoration': 'none',
            'color': '#1E90FF',
            'font-size': '18px',
            'font-weight': 'bold'
        }
        self.logout = pn.widgets.Button(name="Log out")
        self.logout.js_on_click(code="""window.location.href = './logout'""")
        super().__init__(**params)

    def _create_link(self, name, url):
        return pn.pane.HTML(f"""<a href="{url}">{name}</a>""", styles=self.styles)

    def __panel__(self):

        self._layout = pn.Column()
        self._layout.extend(
            [self._create_link(name, url) for name, url in self.pages.items()]
        )
        self._layout.append(self.logout)
        return self._layout
