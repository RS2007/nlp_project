from util import * #noqa
from util import get_vocab
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import numpy as np


# Add your import statements here


class InformationRetrieval:
    def __init__(self):
        self.index = None
        self.vocab_map = None
        self.tfidf = None

    def buildIndexScratch(self,docs,docIDs):
        vocab_map, vocab_size = get_vocab(docs)
        self.vocab_map = vocab_map
        self.index = [[0 for _ in  range(vocab_size)] for _ in range(len(docIDs))]
        self.tfidf_vectorizer = None
        self.tfidf_transformer = None


        for (doc, docID) in zip(docs, docIDs):
            for sentence in doc:
                for token in sentence:
                    if token in vocab_map:
                        self.index[docID - 1][vocab_map[token] - 1] += 1


        tfidf_transformer = TfidfTransformer(
            smooth_idf=True,
            sublinear_tf=True,
        )
        tfidf_matrix = tfidf_transformer.fit_transform(self.index)
        self.tfidf = tfidf_matrix
        self.tfidf_vectorizer = tfidf_transformer


    def buildIndexSK(self,docs,docIDs):
        vocab_map, vocab_size = get_vocab(docs)
        self.vocab_map = vocab_map
        self.index = [[0 for _ in range(len(docIDs))] for _ in range(vocab_size)]
        self.tfidf_vectorizer = None

        for (doc, docID) in zip(docs, docIDs):
            for sentence in doc:
                for token in sentence:
                    if token in vocab_map:
                        self.index[vocab_map[token] - 1][docID - 1] += 1


        docs_strings_per_doc = ['' for _ in range(len(docs))]

        for (doc, docID) in zip(docs, docIDs):
            for sentence in doc:
                sent_accum = ""
                for token in sentence:
                    sent_accum+=token
                    sent_accum+=' '
                sent_accum = sent_accum.strip()
                docs_strings_per_doc[docID-1]+=sent_accum

        tfidf_vectorizer = TfidfVectorizer()
        self.tfidf = tfidf_vectorizer.fit_transform(docs_strings_per_doc)
        words = tfidf_vectorizer.get_feature_names_out()
        self.tfidf_vectorizer = tfidf_vectorizer
        vocab_map2 = {}
        counter = 1
        for word in words:
            vocab_map2[word] = counter
            counter+=1
            
        self.vocab_map = vocab_map2

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
        self.buildIndexSK(docs, docIDs)

    def get_vec(self, query):
        for word in query:
            if word in self.vocab_map:
                self.index[self.vocab_map[word] - 1]

    def rankWithSK(self,queries):
        doc_IDs_ordered = []
        joined_queries = ['' for _ in queries]

        for (id, query) in enumerate(queries):
            for sentence in query:
                sent_accum = ""
                for token in sentence:
                    sent_accum+=token
                    sent_accum+=' '
                sent_accum = sent_accum.strip()
                joined_queries[id]+=sent_accum

        for (query_indx,query) in enumerate(queries):
            query_vec = [0 for _ in range(len(self.vocab_map))]
            for sentence in query:
                for token in sentence:
                    if token in self.vocab_map:
                        query_vec[self.vocab_map[token] - 1] += 1

            query_vec = self.tfidf_vectorizer.transform(joined_queries).toarray()[query_indx]


            query_vec = np.array(query_vec)
            matrix_product = self.tfidf @ query_vec
            for i, row in enumerate(self.tfidf):
                assert np.linalg.norm(query_vec) != 0, "Query vec should not have zero magnitude"
                if np.linalg.norm(row.toarray()) == 0:
                    matrix_product[i] = 0.
                    continue
                matrix_product[i] = matrix_product[i] / (
                    np.linalg.norm(query_vec) * np.linalg.norm(row.toarray())
                )
            doc_IDs_ordered.append(np.argsort(matrix_product)[::-1])

        return doc_IDs_ordered

    def rankScratch(self,queries):
        doc_IDs_ordered = []
        joined_queries = ['' for _ in queries]

        for (id, query) in enumerate(queries):
            for sentence in query:
                sent_accum = ""
                for token in sentence:
                    sent_accum+=token
                    sent_accum+=' '
                sent_accum = sent_accum.strip()
                joined_queries[id]+=sent_accum

        for (query_indx,query) in enumerate(queries):
            query_vec = [0 for _ in range(len(self.vocab_map))]
            for sentence in query:
                for token in sentence:
                    if token in self.vocab_map:
                        query_vec[self.vocab_map[token] - 1] += 1



            query_vec = np.array(query_vec)
            matrix_product = self.tfidf @ query_vec

            for i, row in enumerate(self.tfidf):
                assert np.linalg.norm(query_vec) != 0, "Query vec should not have zero magnitude"
                if np.linalg.norm(row.toarray()) == 0:
                    matrix_product[i] = 0.
                    continue
                matrix_product[i] = matrix_product[i] / (
                    np.linalg.norm(query_vec) * np.linalg.norm(row.toarray())
                )
            doc_IDs_ordered.append(np.argsort(matrix_product)[::-1])


        return doc_IDs_ordered


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

        return self.rankWithSK(queries)
