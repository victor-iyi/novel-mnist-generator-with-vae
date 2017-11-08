"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 02 November, 2017 @ 11:24 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import os
import sys
import pickle
import datetime as dt

import numpy as np


# Base `Dataset` class
class Dataset(object):
    def __init__(self, data_dir, **kwargs):
        """
        Dataset pre-processing class
        :param data_dir:
            top level directory where data resides
        :param kwargs:
            `logging`: Feedback on background metrics
        """
        self._data_dir = data_dir
        # Keyword arguments
        self._logging = kwargs['logging'] if 'logging' in kwargs else True
        # Computed for self.next_batch
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def create(self):
        """Create datasets"""
        self._process()
        self._num_examples = self._X.shape[0]

    def save(self, save_file, force=False):
        """
        Saves the dataset object

        :param save_file: str
            path to a pickle file
        :param force: bool
            force saving
        """
        if os.path.isfile(save_file) and not force:
            raise FileExistsError('{} already exist. Set `force=True` to override.'.format(save_file))
        dirs = save_file.split('/')
        if len(dirs) > 1 and not os.path.isdir('/'.join(dirs[:-1])):
            os.makedirs('/'.join(dirs[:-1]))
        with open(save_file, mode='wb') as f:
            pickle.dump(self, f)

    def load(self, save_file):
        """
        Load a saved Dataset object

        :param save_file:
            path to a pickle file
        :return: obj:
            saved instance of Dataset
        """
        if not os.path.isfile(save_file):
            raise FileNotFoundError('{} was not found.'.format(save_file))
        with open(save_file, 'rb') as f:
            self = pickle.load(file=f)
        return self

    def next_batch(self, batch_size, shuffle=True):
        """
        Get the next batch in the dataset

        :param batch_size: int
            Number of batches to be retrieved
        :param shuffle: bool
            Randomly shuffle the batches returned
        :return:
            Returns `batch_size` batches
            features - np.array([batch_size, ?])
            labels   - np.array([batch_size, ?])
        """
        start = self._index_in_epoch
        # Shuffle for first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            permute = np.arange(self._num_examples)
            np.random.shuffle(permute)
            self._X = self._X[permute]
            self._y = self._y[permute]
        # Go to next batch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_examples = self._num_examples - start
            rest_features = self._X[start:self._num_examples]
            rest_labels = self._y[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                permute = np.arange(self._num_examples)
                np.random.shuffle(permute)
                self._X = self._X[permute]
                self._y = self._y[permute]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_examples
            end = self._index_in_epoch
            features = np.concatenate((rest_features, self._X[start:end]), axis=0)
            labels = np.concatenate((rest_labels, self._y[start:end]), axis=0)
            return features, labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._y[start:end]

    def train_test_split(self, test_size=0.1, **kwargs):
        """
        Splits dataset into training and testing set.

        :param test_size: float, default 0.1
                    Size of the testing data in %.
                    Default is 0.1 or 10% of the dataset.
        :keyword valid_portion: float, None, default
                    Size of validation set in %.
                    This will be taking from training set
                    after splitting into training and testing set.
        :return:
            np.array of [train_X, train_y, test_X, test_y] if
            `valid_portion` is not set
            or
            np.array of [train_X, train_y, test_X, test_y, val_X, val_y] if
            `valid_portion` is set
        """
        test_size = int(len(self._X) * test_size)

        train_X = self._X[:-test_size]
        train_y = self._y[:-test_size]
        test_X = self._X[-test_size:]
        test_y = self._y[-test_size:]

        if 'valid_portion' in kwargs:
            valid_portion = kwargs['valid_portion']
            valid_portion = int(len(train_X) * valid_portion)

            train_X = train_X[:-valid_portion]
            train_y = train_y[:-valid_portion]
            val_X = train_X[-valid_portion:]
            val_y = train_y[-valid_portion:]
            return np.array([train_X, train_y, test_X, test_y, val_X, val_y])

        return np.array([train_X, train_y, test_X, test_y])

    @property
    def features(self):
        return self._X

    @property
    def labels(self):
        return self._y

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_classes(self):
        return self._y.shape[-1]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _process(self):
        pass

    def _one_hot(self, arr):
        arr, uniques = list(arr), list(set(arr))
        encoding = np.zeros(shape=[len(arr), len(uniques)], dtype=np.int32)
        for i, a in enumerate(arr):
            encoding[i, uniques.index(a)] = 1.
        return encoding


# !-------------------------------------- Image Dataset --------------------------------------! #
# `ImageDataset` class for image datasets
class ImageDataset(Dataset):
    def __init__(self, data_dir, grayscale=False, flatten=False, size=50, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.grayscale = grayscale
        self.flatten = flatten
        self.size = size

        self._labels = [l for l in os.listdir(self._data_dir) if l[0] is not '.']
        try:
            from PIL import Image
        except Exception as e:
            raise ModuleNotFoundError('{}'.format(e))
        # First image
        img_dir = os.path.join(self._data_dir, self._labels[0])
        img_file = os.path.join(img_dir, os.listdir(img_dir)[1])
        img = self.__create_image(img_file, return_obj=True)
        self._channel = img.im.bands
        # free memory
        del img_dir
        del img_file
        del img

    @property
    def images(self):
        return self._X

    @property
    def channel(self):
        return self._channel

    def _process(self):
        img_dirs = [os.path.join(self._data_dir, l) for l in self._labels]
        total_images = sum([len(os.listdir(d)) for d in img_dirs])
        if self.flatten:
            self._X = np.zeros(shape=[total_images, self.size * self.size * self.channel])
        else:
            self._X = np.zeros(shape=[total_images, self.size, self.size, self.channel])
        self._y = np.zeros(shape=[total_images, len(self._labels)])
        # Free memory
        del total_images
        del img_dirs
        counter = 0
        for i, label in enumerate(self._labels):
            image_dir = os.path.join(self._data_dir, label)
            image_list = [d for d in os.listdir(image_dir) if d[0] is not '.']
            for j, file in enumerate(image_list):
                try:
                    image_file = os.path.join(image_dir, file)
                    img = self.__create_image(image_file)
                    hot_label = self.__create_label(label)
                    self._X[counter, :] = img
                    self._y[counter, :] = hot_label
                except Exception as e:
                    sys.stderr.write('{}'.format(e))
                    sys.stderr.flush()
                finally:
                    counter += 1
                if self._logging:
                    sys.stdout.write('\rProcessing {} of {} class labels & {} of {} images'.format(
                        i + 1, len(self._labels), j + 1, len(image_list)))
        # Free up memory
        del counter

    def __create_image(self, file, return_obj=False):
        try:
            from PIL import Image
        except Exception as e:
            raise ModuleNotFoundError('{}'.format(e))
        img = Image.open(file)
        img = img.resize((self.size, self.size))
        if self.grayscale:
            img = img.convert('L')
        if return_obj:
            return img
        # convert to np.array
        img = np.array(img, dtype=np.float32)
        if self.flatten:
            img = img.flatten()
        return img

    def __create_label(self, label):
        hot = np.zeros(shape=[len(self._labels)], dtype=int)
        hot[self._labels.index(label)] = 1
        return hot

# !-------------------------------------- Text Dataset --------------------------------------! #
# `TextDataset` for textual dataset
class TextDataset(Dataset):
    def __init__(self, data_dir, window=2, max_word=None, **kwargs):
        """
        Dataset class for pre-processing textual data

        :param data_dir: str
        :param window: int
            is the maximum distance between the current and predicted
            word within a sentence
        :param max_word: int
            Maximum number of words to be kept
        :param kwargs:
        """
        super().__init__(data_dir, **kwargs)
        self._window = window
        self._max_word = max_word

        # TODO: Look into `data_dir`. You may wanna get all files in there and read as a BIG corpus
        corpus_text = open(self._data_dir, mode='r', encoding='utf-8').read()
        if self._max_word:
            corpus_text = corpus_text[:self._max_word]
        corpus_text = corpus_text.lower()
        try:
            from nltk import word_tokenize, sent_tokenize
        except Exception as e:
            raise ModuleNotFoundError('{}'.format(e))
        # word2id & id2word
        unique_words = set(word_tokenize(corpus_text))
        self._vocab_size = len(unique_words)
        self._word2id = {w: i for i, w in enumerate(unique_words)}
        self._id2word = {i: w for i, w in enumerate(unique_words)}

        # Sentences
        raw_sentences = sent_tokenize(corpus_text)
        self._sentences = [word_tokenize(sent) for sent in raw_sentences]

        # Free some memory
        del corpus_text
        del unique_words
        del raw_sentences

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def word2id(self):
        return self._word2id

    @property
    def id2word(self):
        return self._id2word

    @property
    def sentences(self):
        return self._sentences

    def _process(self):
        # Creating features & labels
        self._X = np.zeros(shape=[len(self._sentences), self._vocab_size])
        self._y = np.zeros(shape=[len(self._sentences), self._vocab_size])

        start_time = dt.datetime.now()
        for s, sent in enumerate(self._sentences):
            for i, word in enumerate(sent):
                start = max(i - self._window, 0)
                end = min(self._window + i, len(sent)) + 1
                word_window = sent[start:end]
                for context in word_window:
                    if context is not word:
                        # data.append([word, context])
                        self._X[s] = self._one_hot(self._word2id[word])
                        self._y[s] = self._one_hot(self._word2id[context])
            if self._logging:
                sys.stdout.write(
                    '\rProcessing {:,} of {:,} sentences. Time taken: {}'.format(s + 1, len(self._sentences),
                                                                                 dt.datetime.now() - start_time))
        # Free memory
        del start_time

    def _one_hot(self, idx):
        temp = np.zeros(shape=[self._vocab_size])
        temp[idx] = 1.
        return temp


if __name__ == '__main__':
    data_dir = 'datasets/flowers'
    save_file = 'datasets/saved/features-{0}x{0}.pkl'

    data = ImageDataset(data_dir=data_dir)
    data.create()  # creates features & label
    data.save(save_file.format(data.size))  # saves this object
    # data = data.load(save_file.format(data.size))  # loads saved object

    # Split into training, testing & validation set.
    X_train, y_train, X_test, y_test, X_val, y_val = data.train_test_split(test_size=0.2, valid_portion=0.1)
    # X_train, y_train, X_test, y_test = data.train_test_split(test_size=0.2)

    print('\nTrain: X{}\tTest: y{}\tValid: X{}'.format(X_train.shape, y_test.shape, X_val.shape))
