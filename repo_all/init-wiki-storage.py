#!/usr/bin/python3
"""Validate the deployment's Wiki directory without changing its contents."""
import os
import pathlib


def initialize(library=pathlib.Path('/data/library')):
    if not library.is_dir():
        raise RuntimeError('Wiki library must already exist as a directory: ' + str(library))


if __name__ == '__main__':
    if os.environ.get('WIKI_LIBRARY_ROOT', '/data/library') != '/data/library':
        raise RuntimeError('Wiki v2 uses /data/library')
    initialize()
