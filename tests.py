import unittest
from informationRetrieval import InformationRetrieval
import pandas as pd
from util import get_vocab


class TestInvertedIndex(unittest.TestCase):
    docs = [
        [
            ["this", "is", "the", "first", "document"],
            ["this", "is", "the", "second", "sentence"],
        ],
        [["this", "document", "is", "the", "third", "one."]],
        [["and", "this", "is", "the", "fourth", "document"]],
    ]
    queries = [[["fourth","document"]]]
    docIDs = [1, 2, 3]
    ir = InformationRetrieval()
    ir.buildIndex(docs, docIDs)
    print(pd.DataFrame(ir.index,columns=docIDs,index=get_vocab(docs)[0].keys()))
    print(pd.DataFrame(ir.tfidf.toarray(),columns=docIDs,index=get_vocab(docs)[0].keys()))
    print(ir.rank(queries))
