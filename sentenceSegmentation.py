from util import *

# Add your import statements here
import nltk


class SentenceSegmentation:
    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
                A string (a bunch of sentences)

        Returns
        -------
        list
                A list of strings where each string is a single sentence
        """

        segmentedText = []
        currentSentenceCharBuffer = []
        # TODO: Currently supports only one character delimiters,modify state machine to allow for multiple ones(regex??)
        sentenceSeperators = [".", "!", "?"]
        for char in text:
            if char in sentenceSeperators:
                currentSentenceCharBuffer.append(char)
                segmentedText.append("".join(currentSentenceCharBuffer).strip())
                currentSentenceCharBuffer = []
            else:
                currentSentenceCharBuffer.append(char)

        return segmentedText

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
                A string (a bunch of sentences)

        Returns
        -------
        list
                A list of strings where each strin is a single sentence
        """
        nltk.download("punkt", quiet=True)
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        segmentedText = tokenizer.tokenize(text)
        return segmentedText
