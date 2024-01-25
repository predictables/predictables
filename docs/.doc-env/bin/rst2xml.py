#!/home/aweaver/work/predictables/docs/.doc-env/bin/python

# $Id: rst2xml.py 9115 2022-07-28 17:06:24Z milde $
# Author: David Goodger <goodger@python.org>
# Copyright: This module has been placed in the public domain.

"""
A minimal front end to the Docutils Publisher, producing Docutils XML.
"""

try:
    import locale

    locale.setlocale(locale.LC_ALL, "")
except Exception:
    pass

from docutils.core import default_description, publish_cmdline

description = (
    "Generates Docutils-native XML from standalone "
    "reStructuredText sources.  " + default_description
)

publish_cmdline(writer_name="xml", description=description)
