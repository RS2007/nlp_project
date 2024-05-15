from util import * #noqa
from util import get_vocab
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
import numpy as np
from gensim import corpora, models,similarities
from tqdm import tqdm
import time


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


    def buildIndexForTitle(self,docsBody,docTitles,docIDs,ratio=3,unigram=False):
        docs = [docTitle*ratio +docBody for (docBody,docTitle) in zip(docsBody,docTitles)]
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

        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                           ngram_range=(1,1) if unigram else (2,2),
                                           # max_df=1.,
                                           # min_df=0.01,
                                           sublinear_tf=True
                                           )
        self.tfidf = tfidf_vectorizer.fit_transform(docs_strings_per_doc)
        words = tfidf_vectorizer.get_feature_names_out()
        self.tfidf_vectorizer = tfidf_vectorizer
        vocab_map2 = {}
        counter = 1
        for word in words:
            vocab_map2[word] = counter
            counter+=1
            
        self.vocab_map = vocab_map2



    def buildIndexBigram(self,docs,docIDs):
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

        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                           ngram_range=(1,2),
                                           max_df=1.,
                                           min_df=0.01,
                                           sublinear_tf=True
                                           )
        self.tfidf = tfidf_vectorizer.fit_transform(docs_strings_per_doc)
        words = tfidf_vectorizer.get_feature_names_out()
        self.tfidf_vectorizer = tfidf_vectorizer
        vocab_map2 = {}
        counter = 1
        for word in words:
            vocab_map2[word] = counter
            counter+=1
            
        self.vocab_map = vocab_map2


    def buildBM25WithTitle(self,docsBody,docTitles,ratio=3,k1=1.5,b=0.75):
        documents = [docTitle*ratio +docBody for (docBody,docTitle) in zip(docsBody,docTitles)]
        document_token_arr = [[] for _ in documents]
        for (i,document) in enumerate(documents):
            token_accum = []
            for sentence in document:
                token_accum += sentence
            document_token_arr[i] = token_accum

        dictionary = corpora.Dictionary(document_token_arr)
        # corpus = [dictionary.doc2bow(text) for text in texts]
        bm25Model = models.OkapiBM25Model(dictionary=dictionary)
        bm25_corpus = bm25Model[list(map(dictionary.doc2bow, document_token_arr))]
        bm25_index = similarities.SparseMatrixSimilarity(bm25_corpus, num_docs=len(document_token_arr), num_terms=len(dictionary),
                                   normalize_queries=False, normalize_documents=False)


        self.bm25matrix = bm25_index
        self.bm25dict = dictionary

    def buildBM25(self,documents,k1=1.5,b=.75):
        document_token_arr = [[] for _ in documents]
        print(documents[0])
        for (i,document) in enumerate(documents):
            token_accum = []
            for sentence in document:
                token_accum += sentence
            document_token_arr[i] = token_accum



        dictionary = corpora.Dictionary(document_token_arr)
        # corpus = [dictionary.doc2bow(text) for text in texts]
        bm25Model = models.OkapiBM25Model(dictionary=dictionary)
        bm25_corpus = bm25Model[list(map(dictionary.doc2bow, document_token_arr))]
        bm25_index = similarities.SparseMatrixSimilarity(bm25_corpus, num_docs=len(document_token_arr), num_terms=len(dictionary),
                                   normalize_queries=False, normalize_documents=False)

        self.bm25matrix = bm25_index
        self.bm25dict = dictionary



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

        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1,1),
            max_df=1.,
            min_df=0.01,
            sublinear_tf=True
        )
        self.tfidf = tfidf_vectorizer.fit_transform(docs_strings_per_doc)
        words = tfidf_vectorizer.get_feature_names_out()
        self.tfidf_vectorizer = tfidf_vectorizer
        vocab_map2 = {}
        counter = 1
        for word in words:
            vocab_map2[word] = counter
            counter+=1
            
        self.vocab_map = vocab_map2

    def buildIndex(self, docs,docsTitle, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the documents title
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None

        """
        # The first buildIndex function, represents the construction of the normal TF-IDF term-document matrix
        self.buildIndexSK(docs,docIDs)

        # This function builds the BM25 matrix after weighting the titles with a weight of 3(default)
        self.buildBM25WithTitle(docs,docsTitle)

        # Builds a count term-document matrix, for the purpose of doing LSA 
        self.buildLSAIndex(docs,docIDs)

    def get_vec(self, query):
        for word in query:
            if word in self.vocab_map:
                self.index[self.vocab_map[word] - 1]


    def buildLSAIndex(self,docs,docIDs):
        docs_strings_per_doc = ['' for _ in range(len(docs))]

        for (doc, docID) in zip(docs, docIDs):
            for sentence in doc:
                sent_accum = ""
                for token in sentence:
                    sent_accum+=token
                    sent_accum+=' '
                sent_accum = sent_accum.strip()
                docs_strings_per_doc[docID-1]+=sent_accum
        count_vectorizer = CountVectorizer()
        count_index_matrix = count_vectorizer.fit_transform(docs_strings_per_doc)
        self.count_index_matrix = count_index_matrix.toarray()
        self.count_vectorizer = count_vectorizer


    def scoresWithBM(self,queries):
        import numpy as np
        from gensim.models import TfidfModel
        bm25_results = []
        for query in queries:
            tfidf_model = TfidfModel(dictionary=self.bm25dict, smartirs='bnn')  # Enforce binary weighting of queries
            tfidf_query = tfidf_model[self.bm25dict.doc2bow(' '.join(sum(query,[])).lower().split())]
            similarities = self.bm25matrix[tfidf_query]
            bm25_results.append(similarities)
        return np.array(bm25_results)

    def rankWithBM(self,queries):
        import numpy as np
        from gensim.models import TfidfModel
        bm25_results = []
        for query in queries:
            tfidf_model = TfidfModel(dictionary=self.bm25dict, smartirs='bnn')  # Enforce binary weighting of queries
            tfidf_query = tfidf_model[self.bm25dict.doc2bow(' '.join(sum(query,[])).lower().split())]
            similarities = self.bm25matrix[tfidf_query]
            print(similarities)
            bm25_results.append(np.argsort(similarities)[::-1])
        return np.array([1+doc_ID_ordered for doc_ID_ordered in bm25_results])


    def scoresWithSK(self,queries):
        scores = []
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
                # assert np.linalg.norm(query_vec) != 0, "Query vec should not have zero magnitude"
                if np.linalg.norm(row.toarray()) == 0 or np.linalg.norm(query_vec) == 0:
                    matrix_product[i] = 0.
                    continue
                matrix_product[i] = matrix_product[i] / (
                    np.linalg.norm(query_vec) * np.linalg.norm(row.toarray())
                )
            scores.append(matrix_product)
        return np.array(scores)

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

        return np.array([1+doc_ID_ordered for doc_ID_ordered in doc_IDs_ordered])

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

        return np.array([1+doc_ID_ordered for doc_ID_ordered in doc_IDs_ordered])


    def scoresLSA(self,queries):
        u,s,v = np.linalg.svd(self.count_index_matrix.T)
        s = np.diag(s)
        k = 200 # rank approximation
        u = u[:,:k]
        s = s[:k,:k]
        v = v[:,:k]

        scores = []
        joined_queries = ['' for _ in queries]


        def _sim(x: np.ndarray, y: np.ndarray):
            if (np.linalg.norm(x) * np.linalg.norm(y)) == 0:
                return 0
            return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

        for (id, query) in enumerate(queries):
            for sentence in query:
                sent_accum = ""
                for token in sentence:
                    sent_accum+=token
                    sent_accum+=' '
                sent_accum = sent_accum.strip()
                joined_queries[id]+=sent_accum

        for (query_indx,query) in enumerate(tqdm(queries)):
            query_vec = self.count_vectorizer.transform(joined_queries).toarray()[query_indx]
            q = query_vec.T @ u @ np.linalg.pinv(s)
            d = self.count_index_matrix @ u @ np.linalg.pinv(s)
            res = np.apply_along_axis(lambda row: _sim(q,row),axis=1,arr=d)
            scores.append(-np.min(res)+res)
        return np.array(scores)

    def rankHybrid(self,queries):
        scores_from_LSA = self.scoresLSA(queries)
        scores_from_SK = self.scoresWithBM(queries)
        hybrid = (80*scores_from_SK)/100 + (20*scores_from_LSA)/100
        return np.array([1+np.argsort(doc)[::-1] for doc in hybrid])

    def rankLSA(self, queries):
        u,s,v = np.linalg.svd(self.count_index_matrix.T)
        s = np.diag(s)
        k = 200 # rank approximation
        u = u[:,:k]
        s = s[:k,:k]
        v = v[:,:k]

        doc_IDs_ordered = []
        joined_queries = ['' for _ in queries]


        def _sim(x: np.ndarray, y: np.ndarray):
            if np.linalg.norm(x) * np.linalg.norm(y) == 0:
                return 0
            return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

        for (id, query) in enumerate(queries):
            for sentence in query:
                sent_accum = ""
                for token in sentence:
                    sent_accum+=token
                    sent_accum+=' '
                sent_accum = sent_accum.strip()
                joined_queries[id]+=sent_accum

        for (query_indx,query) in enumerate(tqdm(queries)):
            query_vec = self.count_vectorizer.transform(joined_queries).toarray()[query_indx]
            q = query_vec.T @ u @ np.linalg.pinv(s)
            d = self.count_index_matrix @ u @ np.linalg.pinv(s)
            res = np.apply_along_axis(lambda row: _sim(q,row),axis=1,arr=d)
            ranking = np.argsort(-res)+1
            print("Ranking: ",ranking)
            doc_IDs_ordered.append(ranking)

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

        start_time = time.time_ns()
        ranks = self.rankHybrid(queries)
        end_time = time.time_ns()
        print("Execution time: ", (end_time-start_time) * (10**(-6)), " ms")
        return ranks
