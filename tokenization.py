from util import *

# Add your import statements here




class Tokenization():

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
		tokenizedText = []
		for i in text:
			tokenizedText += [i.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ').replace(';', ' ').split()]

		return tokenizedText



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

		word_tokenizer = TreebankWordTokenizer()
		tokenizedText = []
		for i in text:
			tokenizedText += [word_tokenizer.tokenize(i)]

		return tokenizedText