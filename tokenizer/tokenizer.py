def tokenize(file, method, lang):
    """
    Read a source code file and returns a list of its tokens.

    Attributes
    ----------
    file: str
        Path to the file to be loaded.
    method: str
        Defines the method to be used to generate tokens.
        Acceptable values are: codebert and code_tokenize.
        codebert: Uses the pretrained tokenzier from CodeBERT https://arxiv.org/pdf/2002.08155.pdf.
        code_tokenize: uses the code_tokenize Python library for tokenization.
    lang: str
        Specifies the source code file language.

    Returns
    -------
    tokens: list[str]
        List of tokens.
    """
    tokens = []
    return tokens