from sentenceSegmentation import SentenceSegmentation
from plotter import Plotter
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation

from sys import version_info
import argparse
import json

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:
    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(
            segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", "w")
        )
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(
            tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", "w")
        )
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(
            reducedQueries, open(self.args.out_folder + "reduced_queries.txt", "w")
        )
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(
            stopwordRemovedQueries,
            open(self.args.out_folder + "stopword_removed_queries.txt", "w"),
        )

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", "w"))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", "w"))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", "w"))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(
            stopwordRemovedDocs,
            open(self.args.out_folder + "stopword_removed_docs.txt", "w"),
        )

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", "r"))[:]
        query_ids, queries = (
            [item["query number"] for item in queries_json],
            [item["query"] for item in queries_json],
        )
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        doc_ids, docs,docs_title = (
            [item["id"] for item in docs_json],
            [item["body"] for item in docs_json],
            [item["title"] for item in docs_json],
        )
        # Process documents
        processedDocs = self.preprocessDocs(docs)
        processedDocTitles = self.preprocessDocs(docs_title)


        # Build document index
        # self.informationRetriever.buildIndexSpacy(word_pool(processedDocs),doc_ids)
        # Rank the documents for each query

        # self.informationRetriever.rankLSA(processedQueries)
        # return
        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", "r"))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        # precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        iterations = 10

        # self.informationRetriever.buildIndexForTitle(processedDocs,processedDocTitles, doc_ids)
        # self.informationRetriever.buildBM25WithTitle(processedDocs,processedDocTitles)

        plotter = Plotter()
        # doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
        # plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="vector_space_model_bigram_with_title",iterations=iterations)

        # doc_IDs_ordered = self.informationRetriever.rankWithBM(processedQueries)
        # plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="bm25_model_with_title",iterations=iterations)

        # self.informationRetriever.buildIndexBigram(processedDocs, doc_ids)
        # self.informationRetriever.buildBM25(processedDocs,processedDocTitles)

        # doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
        # plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="vector_space_model_bigram_without_title",iterations=iterations)


        # doc_IDs_ordered = self.informationRetriever.rankWithBM(processedQueries)
        # plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="bm25_model_without_title",iterations=iterations)


        # self.informationRetriever.buildIndexSK(processedDocs, doc_ids)

        # doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
        # plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="vector_space_model_unigram_without_title",iterations=iterations)

        
        # self.informationRetriever.buildIndexForTitle(processedDocs,processedDocTitles, doc_ids,unigram=True)
        # doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
        # plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="vector_space_model_unigram_with_title",iterations=iterations)
        self.informationRetriever.buildIndexSK(processedDocs,doc_ids)
        # self.informationRetriever.buildLSAIndex(processedDocs,doc_ids)
        doc_IDs_ordered = self.informationRetriever.rankWithSK(processedQueries)
        plotter.plot_results(doc_IDs_ordered,query_ids,qrels,args.out_folder,key="",iterations=iterations)






    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        doc_ids, docs,docs_title = (
            [item["id"] for item in docs_json],
            [item["body"] for item in docs_json],
            [item["title"] for item in docs_json],
        )
        # Process documents
        processedDocs = self.preprocessDocs(docs)
        processedTitles = self.preprocessDocs(docs_title)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs,processedTitles, doc_ids)
        # Rank the documents for the query

        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="main.py")

    # Tunable parameters as external arguments
    parser.add_argument(
        "-dataset", default="cranfield/", help="Path to the dataset folder"
    )
    parser.add_argument("-out_folder", default="output/", help="Path to output folder")
    parser.add_argument(
        "-segmenter", default="punkt", help="Sentence Segmenter Type [naive|punkt]"
    )
    parser.add_argument("-tokenizer", default="ptb", help="Tokenizer Type [naive|ptb]")
    parser.add_argument(
        "-custom", action="store_true", help="Take custom query as input"
    )

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    # Either handle query from user or evaluate on the complete dataset
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
