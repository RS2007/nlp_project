from util import * #noqa
from util import get_vocab
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


# Add your import statements here


class InformationRetrieval:
    def __init__(self):
        self.index = None
        self.vocab_map = None
        self.tfidf = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        vocab_map, vocab_size = get_vocab(docs)
        self.vocab_map = vocab_map
        self.index = [[0 for _ in range(len(docIDs))] for _ in range(vocab_size)]

        # Fill in code here
        for doc, docID in zip(docs, docIDs):
            for sentence in doc:
                for token in sentence:
                    if token in vocab_map:
                        self.index[vocab_map[token] - 1][docID - 1] += 1

        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(self.index)
        self.tfidf = tfidf_matrix

    def get_vec(self, query):
        for word in query:
            if word in self.vocab_map:
                self.index[self.vocab_map[word] - 1]

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []
        for query in queries:
            query_vec = [0 for _ in range(len(self.vocab_map))]
            for sentence in query:
                for token in sentence:
                    for token in self.vocab_map:
                        query_vec[self.vocab_map[token] - 1] += 1

            query_vec = np.array(query_vec)
            matrix_product = query_vec @ self.tfidf
            for i, col in enumerate(self.tfidf.T):
                assert np.linalg.norm(query_vec) != 0, "Query vec should not have zero magnitude"
                if np.linalg.norm(col.toarray()) == 0:
                    matrix_product[i] = 0.
                    continue
                matrix_product[i] = matrix_product[i] / (
                    np.linalg.norm(query_vec) * np.linalg.norm(col.toarray())
                )
            doc_IDs_ordered.append(np.argsort(matrix_product)[::-1])

        # Fill in code here

        return doc_IDs_ordered
