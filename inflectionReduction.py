from util import *

# Add your import statements here
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk


class InflectionReduction:
    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of
                stemmed/lemmatized tokens representing a sentence
        """
        nltk.download("wordnet", quiet=True)
        sentenceList = []
        for sentence in text:
            wordList = []
            for word in sentence:
                stemmer = PorterStemmer()
                x = stemmer.stem(word)
                lemmatizer = WordNetLemmatizer()
                x = lemmatizer.lemmatize(x, pos="n")
                x = lemmatizer.lemmatize(x, pos="v")
                wordList.append(x)
            sentenceList.append(wordList)

        # Fill in code here

        return sentenceList
