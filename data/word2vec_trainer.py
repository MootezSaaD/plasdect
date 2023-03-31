import argparse
from gensim.models import Word2Vec
import os
import glob
import multiprocessing
import nltk
nltk.download('punkt')


def read_file(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return file.read()
    
def load_files_by_ext(path, extension='*', encoding='utf8'):
    dir = os.path.join(path, '**',f'*.{extension}')
    return list(glob.iglob(dir, recursive=True))

def train(args):
    files = load_files_by_ext(args.data_path, 'symb')
    sentences = []
    for f in files:
        data = read_file(f)
        tokens = nltk.word_tokenize(data)
        sentences.append(tokens)
    wvmodel = Word2Vec(sentences, min_count=args.min_occ, workers=8, vector_size=args.embedding_size, window=args.window_size, sg=1)
    print('Embedding Size : ', wvmodel.vector_size)
    wvmodel.train(sentences, total_examples=len(sentences), epochs=150)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    save_file_path = os.path.join(args.save_model_dir, args.model_name)
    wvmodel.save(save_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-bin', '--save_model_dir', type=str, default='wv_models')
    parser.add_argument('-n', '--model_name', type=str, default='code')
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-eb', '--embedding_size', type=int, default=128)
    parser.add_argument('-ws', '--window_size', type=int, default=5)
    args = parser.parse_args()
    train(args)
