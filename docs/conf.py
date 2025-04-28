# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Beyond the Field: Revolutionizing Football News Analytics with a Multi-Stage NLP Pipeline'
copyright = '2025, Johns Hopkins University Sports Analytics Research Group'
author = 'Joyce Chen'
release = '4.24.2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.napoleon',
'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'
language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_mock_imports = [
    "llama_index",
    "llama_index.core",
    "llama_index.vector_stores",
    "llama_index.embeddings",
    "llama_index.llms",
    "llama_index.llms.ollama",
    "dotenv",
    "chromadb",
    "langchain_ollama",
]
