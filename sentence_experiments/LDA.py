import torch
import matplotlib
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertConfig
from ast import literal_eval as make_tuple
from argparse import ArgumentParser
import sys
import numpy as np
import pandas as pd
import os, re, random, xlrd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, f1_score


def encode(text, model, tokenizer):
	''' Returns the mean of the last hidden layer as sentence encoding '''
	input_ids = torch.tensor(tokenizer.encode(text.lower(), add_special_tokens=True)).unsqueeze(0)
	outputs = model(input_ids)
	last_layer = torch.squeeze(torch.stack(outputs[2]))
	last_layer_mean = torch.mean(last_layer, dim=0)
	encoded = torch.mean(last_layer_mean, dim=0)
	return encoded


#Load All GermEval
def load_germeval(model, tokenizer, path, subtask=1):
	''' 
	Load the GermEval classes
	if subtask 1: Load all offense and other tweets
	if subtask 2: Load all profanity and other tweets -> sample other down to same size as profanity

	'''
	tweets, labels = [],[]
	with open(path, 'r') as f:
		if subtask == 1:
			for line in f.readlines():
					line = line.strip().split('\t')
					if line[1] == 'OFFENSE':
						tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[0], flags=re.MULTILINE), model, tokenizer).detach().numpy())
						labels.append('OFFENSE')
					elif line[1] == 'OTHER':
						tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[0], flags=re.MULTILINE), model, tokenizer).detach().numpy())
						labels.append('OTHER')
		elif subtask == 2:
			profanity_tweets, other_tweets = [],[]
			for line in f.readlines():
					line = line.strip().split('\t')
					if line[1] == 'OFFENSE' and line[2] == 'PROFANITY':
						profanity_tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[0], flags=re.MULTILINE), model, tokenizer).detach().numpy())
					elif line[1] == 'OTHER':
						other_tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[0], flags=re.MULTILINE), model, tokenizer).detach().numpy())
			other_tweets =  random.sample(other_tweets, len(profanity_tweets)) #downsample OTHER class
			tweets = profanity_tweets + other_tweets
			labels = (['OFFENSE'] * len(profanity_tweets)) + (['OTHER'] * len(other_tweets))
			c = list(zip(tweets, labels))
			random.shuffle(c)
			tweets, labels = zip(*c)

	return tweets, labels


#test on entire GermEval (Subtask 1)
def test_germeval(model, tokenizer, subspace_sentences, subspace_labels, subtask=1):
	''' Fit PCA and LDA to the 46 sentence pairs and then test on GermEval'''
	results = {}
	if subtask == 1:
		germeval_tweets, germeval_labels = load_germeval(model, tokenizer, args.germeval_path)
	elif subtask == 2:
		germeval_tweets, germeval_labels = load_germeval(model, tokenizer, args.germeval_path, subtask=2)

	lda = LDA(n_components=1)
	lda.fit(subspace_sentences, subspace_labels)

	predictions = lda.predict(germeval_tweets)
	results['cm'] = confusion_matrix(y_true=germeval_labels,y_pred=predictions)
	results['f1_micro'] = f1_score(y_true=germeval_labels,y_pred=predictions,average='micro')
	results['f1_macro'] = f1_score(y_true=germeval_labels,y_pred=predictions,average='macro')
	return results

#load and test on hasoc
def load_hasoc(model, tokenizer, path, subtask, encoding=True):
	tweets, labels = [],[]
	with open(path, 'r') as f:
		if subtask == 1:
			for line in f.readlines():
					line = line.strip().split('\t')
					if line[2] == 'HOF':
						tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE))
						labels.append('OFFENSE') if encoding else labels.append('1')
					elif line[2] == 'NOT':
						tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE))
						labels.append('OTHER') if encoding else labels.append('0')
		elif subtask == 2:
			profanity_tweets, other_tweets = [],[]
			for line in f.readlines():
					line = line.strip().split('\t')
					if line[2] == 'HOF' and line[3] == 'PRFN':
						profanity_tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE))
					elif line[2] == 'NOT':
						other_tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[1], flags=re.MULTILINE))
			other_tweets =  random.sample(other_tweets, len(profanity_tweets)) #downsample OTHER class
			tweets = profanity_tweets + other_tweets
			labels = (['OFFENSE'] * len(profanity_tweets)) + (['OTHER'] * len(other_tweets)) if encoding else (['1'] * len(profanity_tweets)) + (['0'] * len(other_tweets))
			c = list(zip(tweets, labels))
			random.shuffle(c)
			tweets, labels = zip(*c)

	return tweets, labels


def test_hasoc(model, tokenizer, subspace_sentences, subspace_labels, language, subtask=1):
	if language == 'de':
		path = args.hasoc_de_path
	elif language == 'en':
		path = args.hasoc_en_path

	results = {}
	if subtask == 1:
		hasoc_tweets, hasoc_labels = load_hasoc(model, tokenizer, path, subtask=1)
	elif subtask == 2:
		hasoc_tweets, hasoc_labels = load_hasoc(model, tokenizer, path, subtask=2)

	lda = LDA(n_components=1)
	lda.fit(subspace_sentences, subspace_labels)

	predictions = lda.predict(hasoc_tweets)
	results['cm'] = confusion_matrix(y_true=hasoc_labels,y_pred=predictions)
	results['f1_micro'] = f1_score(y_true=hasoc_labels,y_pred=predictions,average='micro')
	results['f1_macro'] = f1_score(y_true=hasoc_labels,y_pred=predictions,average='macro')
	return results

#load and test on arab data set
def load_arab(model, tokenizer, path, subtask, encoding=True):
	workbook = xlrd.open_workbook(path)
	sheet = workbook.sheet_by_index(0)
	data = [sheet.row_values(rowx) for rowx in range(sheet.nrows)]
	tweets, labels = [],[]
	if subtask == 1:
		for line in data:
			if line[3] == -1 or line[3] == -2:
				tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE))
				labels.append('OFFENSE') if encoding else labels.append('1')
			elif line[3] == 0:
				tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE))
				labels.append('OTHER') if encoding else labels.append('0')
	elif subtask == 2:
		profanity_tweets, other_tweets = [],[]
		for line in data:
			if line[3] == -2:
				profanity_tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE))
			elif line[3] == 0:
				other_tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[2], flags=re.MULTILINE))
		other_tweets =  random.sample(other_tweets, len(profanity_tweets)) #downsample OTHER class
		tweets = profanity_tweets + other_tweets
		labels = (['OFFENSE'] * len(profanity_tweets)) + (['OTHER'] * len(other_tweets)) if encoding else (['1'] * len(profanity_tweets)) + (['0'] * len(other_tweets))
		c = list(zip(tweets, labels))
		random.shuffle(c)
		tweets, labels = zip(*c)

	return tweets, labels


def test_arab(model, tokenizer, subspace_sentences, subspace_labels, subtask=1):

	results = {}
	if subtask == 1:
		arab_tweets, arab_labels = load_arab(model, tokenizer, args.arab_path, subtask=1)
	elif subtask == 2:
		arab_tweets, arab_labels = load_arab(model, tokenizer, args.arab_path, subtask=2)

	lda = LDA(n_components=1)
	lda.fit(subspace_sentences, subspace_labels)

	predictions = lda.predict(arab_tweets)
	results['cm'] = confusion_matrix(y_true=arab_labels,y_pred=predictions)
	results['f1_micro'] = f1_score(y_true=arab_labels,y_pred=predictions,average='micro')
	results['f1_macro'] = f1_score(y_true=arab_labels,y_pred=predictions,average='macro')

	return results

#load and test on french
def load_french(model, tokenizer, path, encoding=True):
	tweets, labels = [],[]
	with open(path, 'r') as f:
		for line in f.readlines():
			line = line.strip().split('\t')
			if line[4] == "1":
				tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[3], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[3], flags=re.MULTILINE))
				labels.append('OFFENSE') if encoding else labels.append('1')
			elif line[4] == "0":
				tweets.append(encode(re.sub(r'<U\+(.+?)>', '', line[3], flags=re.MULTILINE), model, tokenizer).detach().numpy()) if encoding else tweets.append(re.sub(r'<U\+(.+?)>', '', line[3], flags=re.MULTILINE))
				labels.append('OTHER') if encoding else labels.append('0')
	return tweets, labels


def test_french(model, tokenizer, subspace_sentences, subspace_labels):
	results = {}
	french_tweets, french_labels = load_french(model, tokenizer, args.french_path)

	lda = LDA(n_components=1)
	lda.fit(subspace_sentences, subspace_labels)

	predictions = lda.predict(french_tweets)

	results['cm'] = confusion_matrix(y_true=french_labels,y_pred=predictions)
	results['f1_micro'] = f1_score(y_true=french_labels,y_pred=predictions,average='micro')
	results['f1_macro'] = f1_score(y_true=french_labels,y_pred=predictions,average='macro')

	return results

def main(sentence_corpus, n_sentencepairs):
	config = BertConfig.from_pretrained(args.bert_path, output_hidden_states=True) 
	tokenizer = BertTokenizer.from_pretrained(args.bert_path)
	model = BertModel.from_pretrained(args.bert_path, config=config)
	#load the n sentence pairs to use for the subspace
	pairs_encoded, pairs_decoded = [],[]
	with open("sentence_corpus_filtered","r") as f:
		for line in f:
			pair = make_tuple(line.rstrip())
			pairs_decoded.append(pair)

			offensive_encoded = encode(pair[0], model, tokenizer)
			neutral_encoded = encode(pair[1], model, tokenizer)
			
			pairs_encoded.append((offensive_encoded,neutral_encoded))
	random.shuffle(pairs_encoded)
	pairs_encoded = pairs_encoded[:n_sentencepairs]
	labels = []
	for i in range(len(pairs_encoded)): 
		labels.append('OFFENSE')
		labels.append('OTHER')


	detached_pairs = []
	for a, b in pairs_encoded:
			a,b = a.detach().numpy(), b.detach().numpy()
			detached_pairs.append(a)
			detached_pairs.append(b)
	detached_pairs = np.array(detached_pairs)
	detached_labels = np.array(labels)

	#Germeval
	results_germeval1 = test_germeval(model, tokenizer, detached_pairs, detached_labels)
	print("GermEval Subtask 1 Test:", results_germeval1)
	#test on GermEval Subtask 2 (Profanity vs Other)
	results_germeval2 = test_germeval(model, tokenizer, detached_pairs, detached_labels, subtask=2)
	print("GermEval Subtask 2 Test:", results_germeval2)

	#Hasoc english
	results_hasoc_en1 = test_hasoc(model, tokenizer, detached_pairs, detached_labels, language='en')
	print("Hasoc English Subtask 1 Test:", results_hasoc_en1)
	#test on Hasoc Subtask 2 (Profanity vs Other)
	results_hasoc_en2 = test_hasoc(model, tokenizer, detached_pairs, detached_labels,  language='en',subtask=2)
	print("Hasoc English Subtask 2 Test:", results_hasoc_en2)

	#Arab
	results_arab1 = test_arab(model, tokenizer, detached_pairs, detached_labels)
	print("Arabic Abusive Lang. Detection Subtask 1 Test:", results_arab1)
	#test on Arab Subtask 2 (Profanity vs Other)
	results_arab2 = test_arab(model, tokenizer, detached_pairs, detached_labels, subtask=2)
	print("Arabic Abusive Lang. Detection Subtask 2 Test:", results_arab2)

    #French
    results_french1 = test_french(model, tokenizer, detached_pairs, detached_labels, n_components=n_components)
    print("French Subtask 1 Test:", results_french1)

	return results_germeval1, results_germeval2, results_hasoc_de1, results_hasoc_de2, results_hasoc_en1, results_hasoc_en2, results_arab1, results_arab2, results_french1

if __name__ == "__main__":
	parser = ArgumentParser()

	# Specify the arguments the script will take from the command line
	# Bert model path
	parser.add_argument("-b", "--bert", required=True, dest="bert_path",
						help="Specify path of bert model")
	# Path to Corpus with 100 sentence pairs
	parser.add_argument("-sc", "--sentcorp", required=True, dest="sentence_corpus",
						help="Specify path to corpus with sentence pairs")
	# French test set path
	parser.add_argument("-fr", "--french", required=True, dest="french_path",
						help="Specify path to French test set")
	# Hasoc German test set path
	parser.add_argument("-hd", "--hasocde", required=True, dest="hasoc_de_path",
						help="Specify path to HASOC German test set")
	# Hasoc English test set path
	parser.add_argument("-he", "--hasocen", required=True, dest="hasoc_en_path",
						help="Specify path to HASOC English test set")
	# Arabic Abusive Lang. Detection test set path
	parser.add_argument("-ar", "--arab", required=True, dest="arab_path",
						help="Specify path to rabic Abusive Lang. Detection  test set")
	# Number of sentence pairs to build subspace on
	parser.add_argument("-s", "--sentences", required=True, dest="n_sentences", type=int, 
						help="specify number of sentences to apply LDA on from 1 - 200", choices=list(range(1,201)))


	# Return an argparse object by taking the commands from the command line (using sys.argv)
	args = parser.parse_args() # Argparse returns a namespace object
	main(args.sentence_corpus, args.n_sentences)
