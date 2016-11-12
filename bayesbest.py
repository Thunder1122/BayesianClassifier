# Name: Tushar Chandra (tac311), Trent Cwiok (tbc808), Vyas Alwar (vaa143)
# Date: 5/23/2016
# Description: Best Bayes Classifier
# Improves upon the naive classifier by removing stopwords and adding bigrams (except 
# those with stopwords).
#
# All group members were present and contributing during all work on this project.

import math
import os
import pickle
import re
from random import shuffle
from math import log, e

stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', \
'and', 'any', 'are', 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', \
'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could', 'couldnt', 'did', \
'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', \
'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', \
'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', \
'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', \
'it', 'its', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', \
'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', \
'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', \
'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats', 'the', \
'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', \
'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', \
'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', \
'were', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', \
'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', \
'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves']


class Bayes_Classifier:
	def __init__(self, train_new = 0, data = os.listdir('movies_reviews')):
		"""This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
		cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
		the system will proceed through training.  After running this method, the classifier 
		is ready to classify input text."""

		if os.path.isfile('positive_best.dat') and os.path.isfile('negative_best.dat') and train_new == 0:
			print 'Loading ...'

			self.positive = self.load('positive_best.dat')
			self.negative = self.load('negative_best.dat')

		else:
			self.positive = {'metadata':{'word_count': 0, 'doc_count': 0, 'bigram_count' : 0},
					'words': {}, 'bigrams': {}}

			self.negative = {'metadata':{'word_count': 0, 'doc_count': 0, 'bigram_count' : 0},
					'words': {}, 'bigrams': {}}

			self.train(data)


	def train(self, files_list):
		"""Trains the Naive Bayes Sentiment Classifier. Constructs a frequency dictionary of 
		how often a given word occurs in a positive / negative document. Removes stopwords. """

		print "Training ..."

		for review in files_list:
			# Train on individual words first

			# Positive reviews have a 7th character (rating) of '5'
			positive = review[7] == '5'
			text = self.loadFile('./movies_reviews/' + review)

			# If we have more positive than negative documents, don't train on it
			# This keeps our classifier from being really biased towards positive documents
			if positive:
				if self.positive['metadata']['doc_count'] > self.negative['metadata']['doc_count']:
					continue

			if positive: self.positive['metadata']['doc_count'] += 1
			else: self.negative['metadata']['doc_count'] += 1

			# Build the dictionary of positive / negative word frequencies
			for word in self.tokenize(text):

				# Ignore stopwords
				if word in stopwords:
					continue

				if positive:
					self.positive['metadata']['word_count'] += 1
					try:
						self.positive['words'][word] += 1
					except KeyError:
						self.positive['words'][word] = 1

				else:
					self.negative['metadata']['word_count'] += 1
					try:
						self.negative['words'][word] += 1
					except KeyError:
						self.negative['words'][word] = 1

			# Train on bigrams
			for bigram in self.bigramize(text):
				if positive:
					self.positive['metadata']['bigram_count'] += 1
					try:
						self.positive['bigrams'][bigram] += 1
					except KeyError:
						self.positive['bigrams'][bigram] = 1

				else:
					self.negative['metadata']['bigram_count'] += 1
					try:
						self.negative['bigrams'][bigram] += 1
					except KeyError:
						self.negative['bigrams'][bigram] = 1


		# Keep the bigrams with count of 10 or more only (the vast majority will be 1)
		self.positive['bigrams'] = {k: v for k, v in self.positive['bigrams'].iteritems() if v > 10}
		self.negative['bigrams'] = {k: v for k, v in self.negative['bigrams'].iteritems() if v > 10}

		# Save to file
		self.save(self.positive, 'positive_best.dat')
		self.save(self.negative, 'negative_best.dat')

		return

					
	def classify(self, sText, should_print = False):
		"""Given a target string sText, this function returns the most likely document
		class to which the target string belongs (i.e., positive, negative or neutral).
		Uses a naive Bayesian approach by multiplying all conditional probabilities of 
		a word occurring. We take log of these values to avoid underflow, which also 
		allows us to add the log-probabilities. Ignores stopwords. """

		positive_docs  = float(self.positive['metadata']['doc_count'])
		negative_docs  = float(self.negative['metadata']['doc_count'])
		positive_words = float(self.positive['metadata']['word_count'])
		negative_words = float(self.negative['metadata']['word_count'])
		positive_bigrams = float(self.positive['metadata']['bigram_count'])
		negative_bigrams = float(self.negative['metadata']['bigram_count'])
		
		positive_prob = log(positive_docs / negative_docs)
		negative_prob = log(negative_docs / positive_docs)
		
		# Add probabilities from individual words
		for word in self.tokenize(sText):

			# Ignore stopwords
			if word in stopwords:
				continue

			try:
				cond_positive_freq = self.positive['words'][word] / positive_words 
			except KeyError:
				cond_positive_freq = 1 / positive_words 

			try:
				cond_negative_freq = self.negative['words'][word] / negative_words
			except KeyError:
				cond_negative_freq = 1 / negative_words 
		
			positive_prob += log(cond_positive_freq)
			negative_prob += log(cond_negative_freq)

		# Add probabilities from bigrams
		for bigram in self.bigramize(sText):

			# Ignore stopwords
			if bigram.split()[0] in stopwords or bigram.split()[1] in stopwords:
				continue

			try:
				cond_positive_freq = self.positive['bigrams'][bigram] / positive_bigrams 
			except KeyError:
				cond_positive_freq = 1 / positive_bigrams 

			try:
				cond_negative_freq = self.negative['bigrams'][bigram] / negative_bigrams
			except KeyError:
				cond_negative_freq = 1 / negative_bigrams 
		
			positive_prob += log(cond_positive_freq)
			negative_prob += log(cond_negative_freq)
		
		if should_print:
			print "positive_prob: ", positive_prob
			print "negative_prob: ", negative_prob

		# Within one order of magnitude, classify the document as neutral; otherwise 
		# return the most likely class

		if abs(positive_prob - negative_prob) < 1: return 'neutral'
		if e**positive_prob > e**negative_prob: return 'positive'
		else: return 'negative'


	def test(self, data):
		""" Tests the Bayesian classifier on given data. Uses precision, recall, and f-1 measure
		to evaluate accuracy """

		# Positive correct / incorrect = number of docs correctly / incorrectly assigned positive
		# Negative correct / incorrect = number of docs correctly / incorrectly assigned negative
		positive_correct = 0.0
		positive_incorrect = 0.0
		negative_correct = 0.0
		negative_incorrect = 0.0
		total_positive = 0.0
		total_negative = 0.0

		for review in data:
			text = self.loadFile('./movies_reviews/' + review)

			given_class = self.classify(text)
			real_class = 'positive' if review[7] == '5' else 'negative'

			if given_class == real_class == 'positive': positive_correct += 1
			if given_class == 'positive' and real_class == 'negative': positive_incorrect += 1
			if given_class == real_class == 'negative': negative_correct += 1
			if given_class == 'negative' and real_class == 'positive': negative_incorrect += 1

			if real_class == 'positive': total_positive += 1
			if real_class == 'negative': total_negative += 1

		positive_precision = positive_correct / (positive_correct + positive_incorrect)
		positive_recall = positive_correct / (positive_correct + negative_incorrect)
		negative_precision = negative_correct / (negative_correct + negative_incorrect)
		negative_recall = negative_correct / (negative_correct + positive_incorrect)

		print 'Positive precision = %f' % positive_precision
		print 'Positive recall = %f' % positive_recall
		print 'Negative precision = %f' % negative_precision
		print 'Negative recall = %f' % negative_recall

		print 'Positive F1-measure = %f' % (2 * positive_precision * positive_recall / (positive_precision + positive_recall))
		print 'Negative F1-measure = %f' % (2 * negative_precision * negative_recall / (negative_precision + negative_recall))

		return positive_precision, positive_recall, negative_precision, negative_recall


	def loadFile(self, sFilename):
		"""Given a file name, return the contents of the file as a string."""

		f = open(sFilename, "r")
		sTxt = f.read()
		f.close()
		return sTxt.lower()


	def save(self, dObj, sFilename):
		"""Given an object and a file name, write the object to the file using pickle."""

		f = open(sFilename, "w")
		p = pickle.Pickler(f)
		p.dump(dObj)
		f.close()


	def load(self, sFilename):
		"""Given a file name, load and return the object stored in the file."""

		f = open(sFilename, "r")
		u = pickle.Unpickler(f)
		dObj = u.load()
		f.close()
		return dObj


	def tokenize(self, sText):
		"""Given a string of text sText, returns a list of the individual tokens that 
	    occur in that string (in order). Modified from original to strip capitalization 
	    and punctuation """

		sText = re.sub("[^a-zA-Z0-9 ]", "", sText)
		return sText.lower().split()

	def bigramize(self, sText):
		""" Given a string of text sText, returns a list of the bigrams that occur in 
		that string (in order). Helper function that we added for bigram features."""

		sText = re.sub("[^a-zA-Z0-9 ]", "", sText)
		sTokens = sText.lower().split()

		sBigrams = []
		for i in range(len(sTokens) - 1):
			# Remove bigrams that contain stopwords
			if sTokens[i] in stopwords or sTokens[i + 1] in stopwords:
				continue

			sBigrams.append(sTokens[i] + " " + sTokens[i + 1])

		return sBigrams


def average(L):
	""" Average of a list; simple utility function """
	return sum(L) / float(len(L))


def cross_validate():
	""" 10-fold cross validation"""

	reviews = sorted(os.listdir('movies_reviews'))
	shuffle(reviews)
	fold_size = len(reviews) / 10

	positive_precision = []
	positive_recall = []
	negative_precision = []
	negative_recall = []

	for i in range(10):
		testing = reviews[i * fold_size : (i + 1) * fold_size]
		training = reviews[:i * fold_size] + reviews[(i + 1) * fold_size:]

		c = Bayes_Classifier(train_new = 1, data = training)

		results = c.test(testing)

		positive_precision.append(results[0])
		positive_recall.append(results[1])
		negative_precision.append(results[2])
		negative_recall.append(results[3])

	print 'Average positive precision: %f' % average(positive_precision)
	print 'Average positive recall: %f' % average(positive_recall)
	print 'Average negative precision: %f' % average(negative_precision)
	print 'Average negative recall: %f' % average(negative_recall)	

	print 'Average positive F1-measure = %f' % (2 * average(positive_precision) * average(positive_recall) / (average(positive_precision) + average(positive_recall)))
	print 'Average negative F1-measure = %f' % (2 * average(negative_precision) * average(negative_recall) / (average(negative_precision) + average(negative_recall)))
