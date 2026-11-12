# -*- coding: utf-8 -*-
#
# OMRAT documentation build configuration file
#

import os

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'OMRAT'
copyright = u'2022-2026, Axel Hörteborn / RISE'


# The version is read from the plugin's metadata.txt so the docs never show
# a stale number (it is in the browser tab title: "OMRAT x.y.z documentation").
def _plugin_version():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'metadata.txt')
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('version='):
                    return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return ''


# The full version, including alpha/beta/rc tags.
release = _plugin_version()
# The short X.Y version.
version = '.'.join(release.split('.')[:2])

exclude_patterns = []
pygments_style = 'sphinx'

# -- Options for HTML output ---------------------------------------------------

html_theme = 'alabaster'

html_theme_options = {
    'description': 'Open Maritime Risk Analysis Tool - QGIS Plugin',
    'github_user': 'axelande',
    'github_repo': 'OMRAT',
    'github_button': True,
    'github_type': 'star',
    'fixed_sidebar': True,
    # The whole tree is written into every page; _static/nav_toggle.js folds
    # it and adds the expand / collapse arrows.  Without JavaScript the tree
    # simply stays open.
    'sidebar_collapse': False,
    # Wider sidebar (alabaster default 220px); the page grows by the same
    # amount so the text column keeps its width.  Below page_width alabaster
    # switches to the narrow layout (sidebar on top).
    'sidebar_width': '280px',
    'page_width': '1160px',
}

html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
htmlhelp_basename = 'OMRATdoc'

# -- Custom CSS for source code reference boxes --------------------------------


def setup(app):
    app.add_css_file('custom.css')
    app.add_js_file('nav_toggle.js')


# -- Options for LaTeX output --------------------------------------------------

latex_documents = [
  ('index', 'OMRAT.tex', u'OMRAT Documentation',
   u'Axel Hörteborn', 'manual'),
]

# -- Options for manual page output --------------------------------------------

man_pages = [
    ('index', 'omrat', u'OMRAT Documentation',
     [u'Axel Hörteborn'], 1)
]

# -- Math rendering ------------------------------------------------------------
# Use MathJax for HTML output (no LaTeX installation needed)
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
}

# -- GitHub source code base URL -----------------------------------------------
# Used by RST substitutions for source code links
github_source_url = 'https://github.com/axelande/OMRAT/blob/main'
