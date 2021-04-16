import math
import random
import numpy as np
import tensorflow as tf
import pickle 
import spacy
import json
from nltk.corpus import wordnet
from word2number import w2n
from num2words import num2words
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet as wn

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
nlp2 = spacy.load('en_core_web_lg')
MAX_VOCAB_SIZE = 50000
embedding_matrix = np.load(('/scratch/achaud39/nlp_adversial/nlp_adversarial_examples/aux_files/embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
missed = np.load(('/scratch/achaud39/nlp_adversial/nlp_adversarial_examples/aux_files/missed_embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))

c_ = -2*np.dot(embedding_matrix.T , embedding_matrix)
a = np.sum(np.square(embedding_matrix), axis=0).reshape((1,-1))
b = a.T
dist = a+b+c_
np.save(('/scratch/achaud39/nlp_adversial/nlp_adversarial_examples/aux_files/dist_counter_%d.npy' %(MAX_VOCAB_SIZE)), dist)

with open('/scratch/achaud39/nlp_adversial/nlp_adversarial_examples/aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
	dataset = pickle.load(f)

def get_tags(nlp, q):
	q = q.lower()
	doc = nlp(q) 
	cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = [], [], [], [], []

	for token in doc:
		cap_arr.append(token.text)
		dep_arr.append(token.dep_)
		tag_arr.append(token.tag_)
		pos_arr.append(token.pos_)
	for np in doc.noun_chunks:
		noun_chunks.append(np)
	assert len(cap_arr) == len(dep_arr)
	assert len(cap_arr) == len(tag_arr)
	assert len(cap_arr) == len(pos_arr)
	return cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks

def stem(nlp, a):
	doc = nlp(a)
	a = doc[0]._.lemma()
	return a

def save_json(file_path, data):
	with open(file_path, 'w') as fp:
		json.dump(data, fp, sort_keys=True, indent=4)

def load_json(file_path):
	with open(file_path, 'r') as f: 
		data = json.load(f)
	return data

def save_pickle(file_path, data):
	with open(file_path, 'wb') as handle:
	    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
	with open(file_path, 'rb') as handle:
		return pickle.load(handle)

def compute_dist(subj_emb, all_noun_emb):
	dist_arr = []
	for emb in all_noun_emb:
		diff = np.linalg.norm(subj_emb - emb)
		dist_arr.append(diff)
	dist_arr = np.array(dist_arr)
	return dist_arr


'''
SP HELPER FUNCTIONS
'''

def get_num(num, exactly=False):
	num = num.lower()

	try:
		num_int = int(num)
		is_int = True
	except:
		is_int = False
	eq_num_flag = False
	if is_int:
		eq_num = num2words(num_int)
		eq_num_flag = True
	else:
		try:
			eq_num = str(w2n.word_to_num(num))
			eq_num_flag = True
		except:
			print(num)
			eq_num = 0

	if exactly and eq_num_flag:
		return eq_num

	try:
		num_int = w2n.word_to_num(num)
		digs = int(math.log10(num_int))
	except:
		digs = 1
		num_int = 0

	increament_l= int(math.pow(10, digs))
	increament_h = int(10*math.pow(10, digs))
	new_num_greater_int = random.choice(["less than ", "lesser than ", "maximum ", "at most "]) + num2words(random.choice([random.randrange(num_int + increament_l, num_int+increament_h, increament_l)]))
	if num_int > 0:
		new_num_lower_int = random.choice([ "greater than ", "more than ", "minimum ", "at least "]) + num2words(random.choice([random.randrange(0, num_int, increament_l)]))
		return random.choice([eq_num, new_num_lower_int, new_num_greater_int])
	if eq_num_flag:
		return random.choice([eq_num, new_num_greater_int])	
	return new_num_greater_int

def get_closest_word(word, ret_count=2):
	try:
		src_word = dataset.dict[word]
		word_stem = stem(nlp2, word)
		neighbours, neighbours_dist = pick_most_similar_words(src_word, dist, ret_count, 0.55)
		# print(neighbours, neighbours_dist)
		words = [dataset.inv_dict[x] for x in neighbours]
		# print("words", words)
		results_words = []
		for syn_word in words:
			stem_syn_word = stem(nlp2, syn_word)
			# print(word_stem, stem_syn_word)
			if(stem_syn_word != word_stem):
				results_words.append(syn_word)
		return results_words[:ret_count]
	except:
		return []

def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
	"""
	embeddings is a matrix with (d, vocab_size)
	"""
	dist_order = np.argsort(dist_mat[src_word,:])[1:1+ret_count]
	dist_list = dist_mat[src_word][dist_order]
	# print(dist_list)
	if dist_list[-1] == 0:
		return [], []
	mask = np.ones_like(dist_list)
	if threshold is not None:
		mask = np.where(dist_list < threshold)
		return dist_order[mask], dist_list[mask]
	else:
		return dist_order, dist_list

