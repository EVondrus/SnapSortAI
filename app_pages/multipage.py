import streamlit as st

class MultiPage:
    """
    A class to create a multi-page Streamlit app.

    Parameters:
    app_name (str): The name of the app.

    Attributes:
    pages (list): A list of dictionaries, where each dictionary contains the title and function of a page.
    app_name (str): The name of the app.

    Methods:
    add_page(title, func): Adds a new page to the app.
        title (str): The title of the page.
        func (function): The function to be executed when the page is selected.
    run(): Runs the app.
    """
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title = self.app_name,
            page_icon = "👀",
        )

    def add_page(self, title, func) -> None:
        self.pages.append({'title': title, 'function': func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()