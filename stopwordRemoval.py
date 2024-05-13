from util import *

# Add your import statements here
from nltk.corpus import stopwords
import nltk


class StopwordRemoval:
    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence with stopwords removed
        """

        nltk.download("stopwords", quiet=True)
        stopWordRemovedSentences = []

        stopWords = set(stopwords.words("english"))
        for sentence in text:
            filteredWords = [word for word in sentence if word.lower() not in stopWords]
            stopWordRemovedSentences.append(filteredWords)

        return stopWordRemovedSentences
