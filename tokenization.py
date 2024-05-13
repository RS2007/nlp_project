from util import *

# Add your import statements here
from nltk.tokenize import word_tokenize


class Tokenization:
    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        modifiedWordList = []
        wordDelimiters = [" ", ".", "?", "!","/"]
        for sentence in text:
            tokenizedText = []
            current_token = ""
            for char in sentence:
                if char in wordDelimiters:
                    if current_token:
                        tokenizedText.append(current_token)
                        current_token = ""
                    if char != " ":
                        tokenizedText.append(char)
                else:
                    current_token += char

            if current_token:
                tokenizedText.append(current_token)
            modifiedWordList.append(tokenizedText)

        return modifiedWordList

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        modifiedWordsList = []
        for sentence in text:
            tokenizedText = word_tokenize(sentence)
            modifiedWordsList.append(tokenizedText)


        # Fill in code here

        return modifiedWordsList
