import os
import gensim
import numpy as np
from konlpy.tag import Twitter

twitter = Twitter()
dangerous_keywords = ['[', ']', '(', ')', '{', '}', '\'', '\"', ',', '<', '>', '.', ';', ':']

def rootify(line):
    pos_analyzed = twitter.pos(line)
    return ' '.join(['{}/{}'.format(word, tag) for word, tag in pos_analyzed])


def preprocess(line):
    for keyword in dangerous_keywords:
        line.replace(keyword, ' ')
    return line


def create_corpus():
    file_list = os.listdir('bugs_albums')
    num_files = len(file_list)
    # train_files, test_files = np.split(np.random.permutation(file_list), [int(num_files * 0.9)])
    train_files = np.random.permutation(file_list)

    print('The review files for train : {}'.format(len(train_files)))
    # print('The review files for test : {}'.format(len(test_files)))

    print('Creating train corpus')
    with open('train_corpus_keywords.txt', 'w') as train_corpus_file:
        for trainfile in train_files:
            with open(os.path.join('bugs_albums', trainfile)) as f:
                lines = f.readlines()
                lines = ('{}\n'.format(preprocess(l)) for l in lines)
                train_corpus_file.writelines(lines)

"""
    print('Creating test corpus')
    with open('test_corpus.txt', 'w') as test_corpus_file:
        for testfile in test_files:
            with open(os.path.join('bugs_albums', testfile)) as f:
                lines = f.readlines()
                lines = ('{}\n'.format(l) for l in lines)
                test_corpus_file.writelines(lines)
"""


class SentenceReader:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in open(self.filepath):
            yield line.split(' ')

create_corpus()  # create a corpus text

sentences_vocab = SentenceReader('train_corpus_keywords.txt')
sentences_train = SentenceReader('train_corpus_keywords.txt')

# train!
model = gensim.models.Word2Vec()
model.build_vocab(sentences_vocab)
model.train(sentences_train, total_examples=model.corpus_count, epochs=model.iter)

model.save('model-keywords')

# load model
model = gensim.models.Word2Vec.load('model-keywords')
print(model.most_similar(positive=['따뜻한', '팝'], topn=200))
