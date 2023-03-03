from tree_sitter import Language, Parser
from data import config, language_builder, tree_traverser
from data.ast_parser.comment_deletion import CommentDeletion
from data.ast_parser.var_renaming import VariableRenaming
import os
from utils import io
from tqdm import tqdm

class SymbolParser:
    def __init__(self, language, src, encoding='utf8', kwds = None) -> None:
        self.language = Language(config.LANG_BUILD_PATH, language)
        self.src = src
        self.encoding = encoding
        if kwds:
            self.keywords = io.read_file_lines(kwds, self.encoding)
        else:
            self.keywords = []
        self.load_languages()
        self.parser = Parser()
        self.parser.set_language(self.language)


    def load_languages(self):
        # Build languages
        if not os.path.isfile(config.LANG_BUILD_PATH):
            language_builder.build_langs()

    
    def symbolize(self, in_place=False):
        files = io.load_files_by_ext(self.src, 'code')
        comment_deletion_operator = CommentDeletion(self.parser)
        variable_rename_operator = VariableRenaming(self.parser, keywords=self.keywords)
        for file in tqdm(files):
            file_content = io.read_file(file)
            uncommented_code = comment_deletion_operator.delete_comments(file_content)
            renamed_var_code = variable_rename_operator.rename_variable(uncommented_code)
            if not in_place:
                dst = file.replace('.code', '.symb')
            else:
                dst = file
            io.write_file(dst, renamed_var_code)


# for node in nodes:
#     print(f"{node.type} - {node.text}")
#     print(f"{'=' * 25}")
