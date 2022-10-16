from util import *

# Add your import statements here




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

		lemmatizer = WordNetLemmatizer()
		wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

		def lemmatize_words(token):
			pos_tagged_text = nltk.pos_tag(token.split())
			# print(pos_tagged_text)
			return [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text][0]

		reducedText = []
		for i in text:
			sub_part = []
			for j in i:
				sub_part += [lemmatize_words(j)]
			reducedText += [sub_part]
		
		return reducedText


