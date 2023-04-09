from utils.graph_construction import utc, ifc
from utils.init_features import nltk_tokenizer
from utils.io import read_file
from gensim.models import Word2Vec
from utils.graph_construction import utc, ifc
from data_loader.data_entry import DataEntry
from inference.classifier import PlastDectClassifier
import torch

def inference(w2v_path, src_code, weights, method):
    embeddings = Word2Vec.load(w2v_path).wv
    tokens = nltk_tokenizer(src_code)
    class args:
        window_size = 5

    if method == "ifc":
        adj, features = ifc(tokens, embeddings, args)
    else:
        adj, features = utc(tokens, embeddings, args)
    graph = DataEntry("prod", adj, features, None).graph
    model = PlastDectClassifier.load_from_checkpoint(weights)

    with torch.no_grad():
        y_hat = torch.nn.functional.softmax(model(graph), dim=1)
        
    return y_hat
    