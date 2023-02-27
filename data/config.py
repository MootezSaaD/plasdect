import os, pathlib


BASE = str(pathlib.Path(__file__).resolve().parent.joinpath('tree_sitter_langs'))
JAVA_IMPL = str(os.path.join(BASE,'tree-sitter-java'))
CSHARP_IMPL = str(os.path.join(BASE, 'tree-sitter-c-sharp'))

LANG_BUILD_PATH = str(os.path.join(BASE, 'build', 'languages.so'))
RAW_FILES = str(pathlib.Path(__file__).resolve().parent.joinpath('example_raw'))
KEYWORDS = str(pathlib.Path(__file__).resolve().parent.joinpath('keywords'))