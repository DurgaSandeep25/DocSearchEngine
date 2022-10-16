# Add your import statements here
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pandas as pd
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Add any utility functions here
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

def lemmatize_words(token):

    pos_tagged_text = nltk.pos_tag(token.split())
    # print(pos_tagged_text)
    return [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text][0]

def build_word_index(docs,doc_ids):
    corpora = [] # a list of words
    #word_map = {}
    for doc in docs:
        for sent in doc:
            for word in sent:
                corpora.append(word)
    corpora = set(corpora)  # unique words
    word_map = {word : idx for idx,word in enumerate(set(corpora),0)} # for assigning a unique label to each word

    return word_map
    
# works for both docs and queries
def TF_IDF(docs, doc_ids, word_map, normalize = True):
    """
    docs : list of list of lists
    return :: tf_idf representation of documents (numpy array)
    """
    m = len(set(word_map))  # number of words
    n = len(doc_ids)       # number of docs
    
    tf_idf = np.zeros((m,n)) # initialising

    # filling the tf-idf vector
    for i in range(n):
        for sent in docs[i]:
            for word in sent:
                try:
                    tf_idf[word_map[word]][doc_ids[i]-1] += 1
                except:
                    #print(word)
                    pass
    if (normalize):
        epsilon = 1e-4
        return tf_idf/ (np.linalg.norm(tf_idf, axis = 0)+epsilon)
    # print(tf_idf)
    return tf_idf
	
def Evaluation_metrics(doc_IDs_ordered, query_ids, qrels, n_comp, op_folder = './',save_results = 2, verbose = 1):
    """
    save_results : 0    ===> don't save anything
                 : 1    ===> just save results
                 : > 2  ===> save plots also
    """
    precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
    for k in range(1,11):
        precision = evaluator.meanPrecision(
            doc_IDs_ordered, query_ids, qrels, k)
        precisions.append(precision)
        recall = evaluator.meanRecall(
            doc_IDs_ordered, query_ids, qrels, k)
        recalls.append(recall)
        fscore = evaluator.meanFscore(
            doc_IDs_ordered, query_ids, qrels, k)
        fscores.append(fscore)

        MAP = evaluator.meanAveragePrecision(
            doc_IDs_ordered, query_ids, qrels, k)
        MAPs.append(MAP)
        nDCG = evaluator.meanNDCG(
            doc_IDs_ordered, query_ids, qrels, k)
        nDCGs.append(nDCG)
        if (verbose):
            print("Precision, Recall and F-score @ " +  
                str(k) + " : " + str(precision) + ", " + str(recall) + 
                ", " + str(fscore))
            print("MAP, nDCG @ " +  
                str(k) + " : " + str(MAP) + ", " + str(nDCG))
        if (save_results > 0):
        # saving the results
            with open(op_folder+'Results/LSA_'+str(n_comp)+'.txt', 'a') as f:
                f.write(str(k) + " , " + str(precision) + ", " + str(recall) + 
                        ", " + str(fscore)+", "+str(MAP) + ", " + str(nDCG)+'\n')
            with open(op_folder+'Results/metrics_'+str(k)+'.txt', 'a') as f:
                f.write(str(n_comp) + " , " + str(precision) + ", " + str(recall) + 
                        ", " + str(fscore)+", "+str(MAP) + ", " + str(nDCG)+'\n')
            
    # Plot the metrics and save plot 
    if (save_results > 1):
        plt.figure()
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - LSA "+str(n_comp))
        plt.xlabel("k")
        plt.savefig(op_folder + "Plots/LSA_"+str(n_comp)+".png")