# Graph Neural Networks for Code Smells Detection

## Abstract

Code smells can reveal deep underlying issues in the source code and are often remnants of poor design and implementation of software systems which can lead to issues pertaining to software maintenance and evolution in the future if not detected and refactored on time. Software systems have become increasingly convoluted in the recent years and hence to aid them, the process of software development has also changed drastically, making it more collaborative and distributed across the domain. Adaption of agile based methodology in software development process has expedited the course of making changes to the source code and releasing it to the end user, which further aggravates the need to identify and refactor the source code using code smells. To alleviate this issue, we propose a novel language agnostic approach that uses Graph Neural Network for Code Smell Detection from the source code. During the course of this project we got to experiment with two graph construction methods, Unique Token Construction and Index Focused Construction, to convert code to graphs enabling language-agnostic processing of source code. By utilizing a GNN for classification, these graphs were able to identify class and method specific code smells with promising results. Although more work is needed in the future, these findings offer a promising solution for code smell detection. We believe this paper can be the right step in the direction of having deep learning based language agnostic tools for code smell detection.

## Replication Package

There are two main modules: `data`, and `model`.


## Installing Dependencies

`pip install -r requirements.txt`  
Clone tree-sitter Java and CSharp grammars into `data\tree-sitter-langs`.

### Data Symbolization
- First download the Java or CSharp raw datasets from the link listed at the end README.
- File: `data/symbolize.py`
- Execution: python -m data.symbolize
- Arguments:
  - `src`: root folder containing the four folders that represent each smell. 

### Dataset Preparation
- Modify the `_path` variable in `data/dataset_splitter.py` so that it points to the directory that contains the 4 folders (i.e. `ComplexConditional`, `ComplexMethod`, `FeatureEnvy`, `MultifacetedAbstraction`).
- Run `data/dataset_splitter.py`.
- This will create four CSV files named `<SMELL>.csv`
- Last step, execute `data/strat.py` to perform a 8:1:1 stratified train, validation, test split for each of the generated CSV files:
    - `data_src`: CSV file.
    - `X`: Feature column name (should be set to "file" since it is the column name of `<SMELL>.csv`).
    - `y`: Target column name (should be set to "label" since it is the column name of `<SMELL>.csv`).
    - `output`: name of the output file name

- At the end of this step you should have 4 CSV files for each smell that map each symbolized file to a label.


### Word2Vec Training
- File: `data/word2vec_trainer.py`
- Execution: `python -m data.word2vec_trainer`
- Arguments:
  - `data_path`: corpus_folder. It should be the root folder containing the four folders that represent each smell. 
  - `min_occ`: minimum occurrence of a given token.
  - `epochs`: number of training epochs. 
  - `embedding_size`: size of the embedding vector.
  - `window_size`: context window size.

### GGNN Training
- File: `model/trainer.py`
Execution: `python model/trainer.py`
Arguments:
  - `dataset`: CSV file containing mapping to physical files. Columns: ‘file’, ‘split’, ‘label’. It should be one the CSV files generated in the Dataset Preparation step.
  - `epochs`: number of training epochs. 
  - `feature_size`: node embedding size (i.e. the one specified in Word2Vec training) 
  - `graph_embed_size`: size of the hidden state of the GNN.
  - `build_method`: UTC or IFC.
  - `w2v`: path to the Word2Vec model. 
  - `window_size`: window size to be used by UTC or IFC.

### Inference
- File: `inference/infer.py`.
Execution: `python inference/infer.py`
Arguments:
  - `w2v_path`: Path to a pretrained Word2Vec model.
  - `src_code`: Content of a source code snippet. You can change this by refactoring the script to make it accept a file path instead and use the `read_file` method (already imported).
  - `weights`: Checkpoint file (.cpkt) that contains the model's weights.
  - `method`: Graph construction method

## Resource
- Java tree-sitter grammar: https://github.com/tree-sitter/tree-sitter-java.
- CSharp tree-sitter grammar: https://github.com/tree-sitter/tree-sitter-c-sharp
- Raw Java Dataset (contains 4 subfolders for each smell): https://drive.google.com/file/d/1l8B25M5wmnf8eNURFMpY5Vt4b0w8hGlR/view?usp=sharing
- Raw CSharp Dataset (contains 4 subfolders for each smell): https://drive.google.com/file/d/1jxe3PxFqhIiNE5jn2sl7iz5XuySYMMxe/view?usp=sharing
- Pretrained Word2Vec model on Java Corpus: https://drive.google.com/file/d/19TwT4pTDv08ziFhBZ8XTYoLNP6pXscfC/view?usp=share_link
- Pretrained Word2Vec model on CSharp Corpus: https://drive.google.com/file/d/1SqvpgfRDfGlOznS_JOM2zjFOf-Mrh5GL/view?usp=sharing
- Model weights and logs can be found in `model/weights` folder.