import collections
import os
import tensorflow as tf
import numpy as np
import json


class WordDictionary:
    def __init__(self, base_path, vocab_size=50000):
        self.VOCAB_SIZE = vocab_size
        self.UNKNOWN_WORD = '<unk>'
        self.UNKNOWN_IDX = self.VOCAB_SIZE
        self.base_path = base_path
        self.files = self.get_files()
        self.dictionary, self.reverse_dict = self.build_dictionary(self.files)
        self.skip_gram_pairs = self.build_word_pairs()
        # train / test pairs
        self.test_data, self.train_data = self.split_train_test_data()

    def split_train_test_data(self, ratio=0.15):
        total_len = len(self.skip_gram_pairs)
        test_data, train_data = np.split(np.array(self.skip_gram_pairs),
                                         [int(total_len * ratio)])
        return test_data, train_data

    def build_word_pairs_file(self, filename):
        word_indices = self.word_to_idx_file(filename)
        pairs = self.word_pairs(word_indices, window_size=1)

        # prints for debug - better turn off due to I/O latency
        # print('Created word pairs for file : {}'.format(filename))
        # print(pairs) # debug
        return pairs

    def build_word_pairs(self, word_pair_filename='word_pair.json'):
        if os.path.exists(word_pair_filename):
            print('Reading word pairs...')
            with open(word_pair_filename, 'r') as f:
                pairs = json.load(f)
        else:
            pairs = []
            print('Building word pairs...')
            for filename in self.files:
                pairs.extend(self.build_word_pairs_file(filename))
            print('Writing word pairs at : {}'.format(word_pair_filename))
            with open(word_pair_filename, 'w') as f:
                json.dump(pairs, f, ensure_ascii=False)
            print('Word pair build complete')
        return pairs

    def generate_batch(self, batch_size):
        assert batch_size < len(self.train_data)
        x_data = []
        y_data = []
        rand_idx = np.random.choice(range(len(self.skip_gram_pairs)),
                                    batch_size, replace=False)

        for idx in rand_idx:
            x_data.append(self.skip_gram_pairs[idx][0])
            y_data.append([self.skip_gram_pairs[idx][1]])
        return x_data, y_data

    @staticmethod
    def word_pairs(word_indices, window_size: int):
        sentence_size = len(word_indices)
        pairs_per_word = []
        for idx, word_id in enumerate(word_indices):
            within_window = []
            for i in range(window_size):
                forward_window = idx + (i + 1)
                behind_window = idx - (i + 1)

                if forward_window < sentence_size:
                    within_window.append(word_indices[forward_window])

                if behind_window >= 0:
                    within_window.append(word_indices[behind_window])

            pairs_per_word.append((word_id, within_window))

        pairs = []
        for pairs_word in pairs_per_word:
            for word in pairs_word[1]:
                pairs.append((pairs_word[0], word))

        return pairs

    def word_to_idx_file(self, filename):
        filepath = os.path.join(base_path, filename)
        word_seq = []
        with open(filepath) as f:
            for line in f:
                word_seq.extend(line.strip().split())
        # print(word_seq) # debug

        word_indices = []
        for word in word_seq:
            if word not in self.dictionary:
                word_indices.append(self.UNKNOWN_IDX)  # not in vocabulary
            else:
                word_indices.append(self.dictionary[word])

        # word_indices = [self.dictionary[word] for word in word_seq]
        return word_indices

    def get_files(self):
        print('Retreiving data files from : {}'.format(self.base_path))
        filelist = os.listdir(self.base_path)
        np.random.shuffle(filelist)
        return filelist

    def build_dictionary(self, filelist,
                         dictionary_filename='dictionary.json',
                         reverse_dict_filename='reverse_dict.json'):
        # if the file exists, simply read the file and store as dictionary
        if os.path.exists(dictionary_filename) and os.path.exists(reverse_dict_filename):
            print('Opening dictionary file')
            with open(dictionary_filename, 'r') as dict_json:
                dictionary = json.load(dict_json)
            with open(reverse_dict_filename, 'r') as rev_dict_json:
                reverse_dict = json.load(rev_dict_json)
        else:
            words = list()
            print('The dictionary file does not exist - creating dictionary.')
            print('Reading file...')
            for file in filelist:
                with open(os.path.join(self.base_path, file)) as f:
                    for line in f:
                        words.extend(line.strip().split())
                # print('Reading file : {}'.format(file))

            print('Finished reading all files. Creating word counts:')

            # in order of most comm.
            counter = collections.Counter(words).most_common()
            counter = counter[:self.VOCAB_SIZE]  # cut the counter to limit
            print(counter[:10])  # sample contents

            # construct dictionary
            dictionary = {word[0]: idx for idx, word in enumerate(counter)}
            dictionary[self.UNKNOWN_WORD] = self.UNKNOWN_IDX
            print('The length of the dictionary is : {}'.format(len(dictionary)))

            # construct reverse dictionary : id -> word
            reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
            reverse_dict[self.UNKNOWN_IDX] = self.UNKNOWN_WORD

            print('Saving dictionary to file')
            with open(dictionary_filename, 'w') as dict_json:
                json.dump(dictionary, dict_json, ensure_ascii=False)
            with open(reverse_dict_filename, 'w') as rev_dict_json:
                json.dump(reverse_dict, rev_dict_json, ensure_ascii=False)

        return dictionary, reverse_dict


if __name__ == '__main__':
    base_path = os.path.join(os.getcwd(), 'bugs_albums')
    wdic = WordDictionary(base_path)
    # wdic.word_to_idx_file('266812')
    # wdic.build_word_pairs_file('266812')

    # create network!
    batch_size = 20
    vocab_size = len(wdic.dictionary)
    embedding_size = 100
    num_sampled = 15  # TODO: use?
    valid_size = 32  # random set of words to evaluate similarity
    # choose any 32 words within top 500 words (indexes)
    valid_examples = np.random.choice(500, 32, replace=False)

    train_inputs = tf.placeholder(dtype=tf.int64, shape=[batch_size])
    # need shape [batch_size, 1] for nn.nce_loss
    train_labels = tf.placeholder(dtype=tf.int64, shape=[batch_size, 1])
    # example words to test the similarity on
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # use CPU
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size]))
        # TODO: result size = [batch_size, embedding_size]?
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # construct weights and bias for NCE loss
    nce_weights = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

    # compute the average NCE loss for the batch
    # TODO: NCE == noise-contrastive estimation - num_sampled meaning?
    nce_loss = tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                              num_sampled, vocab_size)
    loss = tf.reduce_mean(nce_loss)
    tf.summary.scalar('loss', loss)  # attach summary for tensorboard usage

    # optimizer
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    print('optimizer ready')

    # SIMILARITY
    # keep_dims makes the result still have 2-D dimension
    # normalize
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # similarity matrix result shape = (valid_size, vocab_size), each row contains similarity
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


    # start session
    with tf.Session() as sess:
        # merge summaries and write out to logs folder
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('logs/', sess.graph)

        sess.run(tf.global_variables_initializer())

        for step in range(5000):
            print('step: {}'.format(step))
            # summary, loss = sess.run([merged, loss], feed_dict=) # TODO
            batch_inputs, batch_labels = wdic.generate_batch(batch_size)
            _, loss_val = sess.run([train_op, loss],
                                   feed_dict={train_inputs: batch_inputs,
                                              train_labels: batch_labels})

            if step % 500 == 0:
                print("Loss at step {} : {}".format(step, loss_val))

            # find nearest words with a specified word
            if step % 1000 == 0:
                similarity_val = sess.run(similarity)
                for idx in range(valid_size):
                    valid_word = wdic.reverse_dict[str(valid_examples[idx])]
                    top_k = 6
                    nearest = (-similarity_val[idx, :]).argsort()[1:top_k + 1]
                    print('Nearest top 6 words to : {}'.format(valid_word))
                    for word_idx in nearest:
                        print('{}'.format(wdic.reverse_dict[str(word_idx)]))

        trained_embeddings = embeddings.eval()
