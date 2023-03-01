import argparse
from data.parser import SymbolParser


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_argument("--src", default=".", help="Dataset path", type=str, required=True)
    language = parser.add_argument("--language", help="Specify dataset language. Options: java, c_sharp", type=str, required=True)
    lang_kws = parser.add_argument("--language-kwds", help="Specify list of keywords file to ignore during symbolization.", type=str, required=False)
    enconding = parser.add_argument("--encoding", default="utf8", help="Specify dataset encoding. Default is utf8.", type=str)
    
    args = parser.parse_args()
    src, language, lang_kws, encoding = args.__dict__.values()
    sym_parser = SymbolParser(language, src, encoding, lang_kws)
    sym_parser.symbolize()


    





if __name__ == "__main__":
    main()