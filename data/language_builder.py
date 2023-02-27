from tree_sitter import Language
from data import config


def build_langs():
    Language.build_library(
        config.LANG_BUILD_PATH,
        [config.JAVA_IMPL, config.CSHARP_IMPL]
    )

