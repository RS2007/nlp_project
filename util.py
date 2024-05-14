import numpy as np
# Add your import statements here


# Add any utility functions here


def orderedSplitAtDelimiter():
    assert False, "Unimplemented"


class WordVector:
    def __init__(self, basis):
        self.index2basis = basis
        self.basis_length = len(basis)
        self.basis2index = {}
        for i, word in enumerate(basis):
            self.basis2index[word] = i

    def get_vec(self, sentence):
        result = [0 for _ in range(self.basis_length)]
        if type(sentence) == "string":
            for word in sentence.split():
                if word in self.index2basis:
                    result[self.basis2index[word]] += 1
            return np.array(result)
        else:
            for word in sentence:
                if word in self.index2basis:
                    result[self.basis2index[word]] += 1
            return np.array(result)



def argmax_all(arr):
    return np.argwhere(arr == np.amax(arr)).reshape(-1)


def cosine_sim(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_vocab(docs):
    seen = {}
    counter = 1
    for doc in docs:
        for sentence in doc:
            for word in sentence:
                if word not in seen:
                    seen[word] = counter
                    counter+=1
    return seen,len(seen)

def word_pool(docs):
    """
    pools all the words in a list for each sentence to a list for each doc for all docs
    :param docs: a list of docs in which each doc is a list of sentences which are lists of words
    :return: a list of docs in which each doc is a list of words
    """
    new_docs = []
    for doc in docs:
        new_doc = []
        for sentence in doc:
            new_doc += sentence
        new_docs.append(new_doc)
    return new_docs
