from utils.graph_construction import utc, ifc
from utils.init_features import nltk_tokenizer
from utils.io import read_file
from gensim.models import Word2Vec
from utils.graph_construction import utc, ifc
from data_loader.data_entry import DataEntry
from inference.classifier import PlastDectClassifier
import torch
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v_path', type=str, required=True, help='Pretrained Word2Vec model.')
    parser.add_argument('--src_code', type=str, required=True, help='Sample src code.')
    parser.add_argument('--weights', type=str, required=True, help='CPKT File.')
    parser.add_argument('--method', type=str, required=True, help='ifc or utc.')

    args = parser.parse_args()

    result = inference(args.w2v_path, args.src_code, args.weights, args.method)

    print(result)
