from util import *

# Add your import statements here
from collections import defaultdict
import numpy as np

class InformationRetrieval():

	def __init__(self):
		self.index = None

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

		index = None

		#Fill in code here

		index = {tokens: [] for d in docs for sentence in d for tokens in sentence}
		for i in range(len(docs)):
			doc = [token for sent in docs[i] for token in sent]
			for j in docs[i]:
				for k in j:
					if k in index.keys():
						if [docIDs[i], doc.count(k)] not in index[k]:
							index[k].append([docIDs[i], doc.count(k)])
		self.index = (index, len(docs), docIDs)


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

		doc_IDs_ordered = []

		#Fill in code here
		index, doc_num , doc_ID = self.index

		D = np.zeros((doc_num, len(index.keys())))
		key = list(index.keys())

		for i in range(len(key)):
			for l in index[key[i]]:
				D[l[0]-1, i] = l[1]

		idf = np.zeros((len(key), 1))
		for i in range(len(key)):
			idf[i] = np.log10(doc_num / (len(index[key[i]])))
		for i in range(D.shape[0]):
			D[i, :] = D[i, :] * idf.T

		for i in range(len(queries)):
			query = defaultdict(list)
			for j in queries[i]:
				for k in j:
					if k in index.keys():
						query[k] = index[k]
			query = dict(query)
			Q = np.zeros((1, len(key)))
			for m in range(len(key)):
				if key[m] in query.keys():
					Q[0, m] = 1

			Q = Q*idf.T

			# cosine similarity
			temp = []
			for d in range(D.shape[0]):
				simi = np.dot(Q[0, :], D[d, :])/((np.linalg.norm(Q[0, :]) + 1e-4)*(np.linalg.norm(D[d, :]) + 1e-4))
				temp.append(simi)
			doc_IDs_ordered .append([x for _, x in sorted(zip(temp, doc_ID), reverse=True)])
		return doc_IDs_ordered


