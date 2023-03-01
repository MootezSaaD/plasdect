"""
Author: Nghi D. Q. Bui
URL: https://github.com/bdqnghi/code_transformation
"""

class BaseOperator():

    def __init__(self, parser):

        self.parser = parser
       
    def parse(self, code_snippet):
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        return tree