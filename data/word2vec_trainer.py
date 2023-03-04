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
    files = load_files_by_ext(args.data_paths, 'code')
    sentences = []
    for f in files:
        data = read_file(f)
        tokens = nltk.word_tokenize(data)
        sentences.append(tokens)
    wvmodel = Word2Vec(sentences, min_count=args.min_occ, workers=multiprocessing.cpu_count(), size=args.embedding_size)
    print('Embedding Size : ', wvmodel.vector_size)
    for i in range(args.epochs):
        wvmodel.train(sentences, total_examples=len(sentences), epochs=1)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    save_file_path = os.path.join(args.save_model_dir, args.model_name)
    wvmodel.save(save_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str, nargs='+', default=['data/code_train.json', 'data/code_test.json'])
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-bin', '--save_model_dir', type=str, default='wv_models')
    parser.add_argument('-n', '--model_name', type=str, default='code')
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    args = parser.parse_args()
    train(args)