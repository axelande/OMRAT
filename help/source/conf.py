# -*- coding: utf-8 -*-
#
# OMRAT documentation build configuration file
#

import sys, os

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'OMRAT'
copyright = u'2022-2026, Axel Hörteborn / RISE'

# The short X.Y version.
version = '0.3'
# The full version, including alpha/beta/rc tags.
release = '0.3.1'

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
    'sidebar_collapse': True,
    'page_width': '1100px',
}

html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
htmlhelp_basename = 'OMRATdoc'

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
