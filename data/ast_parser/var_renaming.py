"""
Author: Nghi D. Q. Bui
URL: https://github.com/bdqnghi/code_transformation
"""
from data.ast_parser.base_operator import BaseOperator


class VariableRenaming(BaseOperator):
    def __init__(self, language: str, keywords = []):
        super(VariableRenaming, self).__init__(language)
        self.var_node_types = {'identifier'}
        self.var_filter_types = {'class_declaration', 'method_declaration', 'method_invocation'}
        self.num_types = {'decimal_floating_point_literal', 'decimal_integer_literal'}
        self.keywords = set(keywords) # To be used when replacing tokens. These are tokens that reserved for each language, and we 
                           # may don't want them to be replaced.

    # Get only variable node from type "identifier"
    def get_identifier_nodes(self, tree, text):
        var_nodes, var_renames = [], {}
        func_nodes, func_renames = [], {}
        string_literal_nodes, string_literal_renames = [], {}
        num_nodes, num_renames = [], {}
        queue = [tree.root_node]
        while queue:
            current_node = queue.pop(0)
            if str(current_node.type) == 'method_declaration':
                # Get its identifier children
                id_children = list(filter(lambda x: x.type == 'identifier', current_node.children))
                func_name = text[id_children[0].start_byte: id_children[0].end_byte]
                if func_name in self.keywords:
                    continue
                if func_name not in func_renames and func_name not in var_renames:
                    func_renames[func_name] = "FUNC{}".format(len(func_renames) + 1)
                func_nodes.append([id_children[0], func_name, func_renames[func_name]])

            elif str(current_node.type) == 'method_invocation':
                children = current_node.children
                if len(children) == 4: # identifier.identifier(argument_list)
                    id1_node = children[0]
                    id1_name = text[id1_node.start_byte: id1_node.end_byte]

                    func_node = children[2]
                    func_name = text[func_node.start_byte: func_node.end_byte]

                    if id1_name in self.keywords or func_name in self.keywords:
                        continue

                    if id1_name not in var_renames:
                        var_renames[id1_name] = "VAR{}".format(len(var_renames) + 1)
                    var_nodes.append([id1_node, id1_name, var_renames[id1_name]])

                    if func_name not in func_renames:
                        func_renames[func_name] = "FUNC{}".format(len(func_renames) + 1)
                    func_nodes.append([func_node, func_name, func_renames[func_name]])

                if len(children) == 2: # identifier(argument_list)
                    func_node = children[0]
                    func_name = text[func_node.start_byte: func_node.end_byte]

                    if func_name in self.keywords:
                        continue

                    if func_name not in func_renames:
                        func_renames[func_name] = "FUNC{}".format(len(func_renames) + 1)
                    func_nodes.append([func_node, func_name, func_renames[func_name]])
          
            for child_node in current_node.children:
                child_type = str(child_node.type)
                if child_type in self.var_node_types and str(current_node.type) not in self.var_filter_types:  # only identifier node and none method related
                    var_name = text[child_node.start_byte: child_node.end_byte]

                    if var_name in self.keywords:
                        continue

                    if var_name not in var_renames:
                        var_renames[var_name] = "VAR{}".format(len(var_renames) + 1)
                    var_nodes.append([child_node, var_name, var_renames[var_name]])
                # Mapping integer to NUMBER
                if child_type in self.num_types:
                    num_node = child_node
                    num_value = text[num_node.start_byte: num_node.end_byte]
                    if num_value not in num_renames:
                        num_renames[num_value] = "NUMBER"
                    num_nodes.append([num_node, num_value, num_renames[num_value]])
                # Mapping string values to STRING
                if child_type == 'string_literal':
                    str_node = child_node
                    str_value = text[str_node.start_byte: str_node.end_byte]
                    if str_value not in string_literal_renames:
                        string_literal_renames[str_value] = "STRING"
                    string_literal_nodes.append([str_node, str_value, string_literal_renames[str_value]])

                queue.append(child_node)

        return var_nodes + func_nodes + num_nodes + string_literal_nodes

    def transform(self, id_nodes, code_text):
        id_nodes = sorted(id_nodes, reverse=True, key=lambda x: x[0].start_byte)
        for var_node, var_name, var_rename in id_nodes:
            code_text = code_text[:var_node.start_byte] + var_rename + code_text[var_node.end_byte:]
        return code_text

    def rename_variable(self, code_snippet):
        tree = self.parse(code_snippet)
        identifier_nodes = self.get_identifier_nodes(tree, code_snippet)
        return self.transform(identifier_nodes, code_snippet)
