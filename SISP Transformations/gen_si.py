import random 
import spacy 
import json 
import sys
import numpy as np 
import pickle
import logging
import nltk
import lemminflect
import sys
import yaml
import math
import time
import copy
import Levenshtein
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import wordnet 
from tqdm import tqdm 
from scipy.spatial import distance
from word2number import w2n
from num2words import num2words
EXCLUDE_LIST = ["photo", "photos", "photograph", "photographs", 
				"picture", "pictures", "image", "images", "show", "shows", "feature", 
				"features", "contain", "contains", "depict", "depicts", "display", "displays", 
				"visible", "displayed", "depicted", "appear", "appears", "none", "right", "left", "interface"]

file = open('config.yaml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# spacy.require_gpu()
nlp = spacy.load('en_core_web_lg')
def append2dict(d, key, val):
	if key in d:
		d[key] += val 
	else: 
		d[key] = val 
cf_counts = dict()
'''
FUNCTIONS FOR INDIVIDUAL DATASETS
'''
def get_violin_si_sentences():
	annotations_file = cfg['files']['violin_annotations']
	res = dict()
	with open(annotations_file, 'r') as f: 
		all_data = json.load(f)
	data = []
	for clip_id, data_item in tqdm(all_data.items()):
		data.append(data_item)

	for data_item in tqdm(data):
		clip_id = data_item['file']
		# logging.info("####### PROCESSING :: {} ########".format(clip_id))
		item_statements = data_item['statement']
		tags_li, stmts_li = [], []
		for statement_pair in item_statements:
			# Using the negative statement to get the cf
			pos_stmt, neg_stmt = statement_pair[0], statement_pair[1]

			sp_pos_stmts, sp_pos_tags = get_cfs_for_pos_statement(pos_stmt)
			sp_neg_stmts, sp_neg_tags = get_cfs_for_neg_statement(neg_stmt)

			tags = [sp_pos_tags, sp_neg_tags]
			stmts = [sp_pos_stmts, sp_neg_stmts]

			tags_li.append(tags)
			stmts_li.append(stmts)

		data_item['si_statements'] = stmts_li
		data_item['si_tags'] = tags_li
		res[clip_id] = data_item

	print("Category-wise count ", cf_counts)
	# print("Total statements ", len(list(res.keys())))
	save_json(cfg['files']['violin_si_annotations'], res)
	# save_json("debug_violin.json", res)

def get_nlvr_si_sentences(split):
	file_path = cfg['files']['nlvr_'+split+'_file']
	with open(file_path, 'r') as f: 
		data = json.load(f)
	cf_res = list()
	data_li = []
	# Using only positive statements
	for item in data:
		item['tag'] = 'orig'
		# if item['label']:
		# 	data_li.append(item)
	
	# count_dict = dict()
	uid_counter = len(data) + 1
	for item in tqdm(data):
		if item['tag'] !=  'orig': continue
		img0_id = item['img0'].replace("-img0", "")
		sent = item['sent']
		if item['label']:
			cf_stmts, cf_tags = get_cfs_for_pos_statement(sent)
		else: 
			cf_stmts, cf_tags = get_cfs_for_neg_statement(sent)
		for i, cf_stmt in enumerate(cf_stmts):
			if cf_stmt is None: continue 
			cf_tag = cf_tags[i]
			new_item = copy.copy(item)
			new_item['sent'] = cf_stmt
			new_item['orig_sent'] = item['sent']
			new_item['tag'] = cf_tag
			new_item['uid'] = "nlvr2_{}_{}".format(split, uid_counter)
			uid_counter += 1
			new_item['label'] = 1 - item['label']
			new_item['orig_label'] = item['label']
			new_item['parent_identifier'] = item['identifier']
			new_item['parent_uid'] = item['uid']
			new_item['identifier'] = "{}-{}".format(img0_id, uid_counter)
			cf_res.append(new_item)
	
	print("Total statements ", len(cf_res))
	cf_res += data
	print(cf_counts)
	save_json(cfg['files']['nlvr_'+split+'_si_file'], cf_res)
	# save_json("debug_si.json", cf_res)

def get_vqa_si_sentences(split):
	print("Inside vqa")
	file_path = cfg['files']['vqa_'+split+'_file']
	with open(file_path, 'r') as f: 
		data = json.load(f)
	cf_res = list()
	data_li = []
	# # Using only positive statements
	for item in data:
		# if item['label']:
		item['tag'] = 'orig'
		data_li.append(item)
	
	# count_dict = dict()
	for item in tqdm(data_li):
		if item['tag'] != 'orig': continue
		img0_id = item['img_id']
		sent = item['sent']
		if 'yes' in item['label']:
			cf_stmts, cf_tags = get_cfs_for_pos_statement(sent)
		else: 
			cf_stmts, cf_tags = get_cfs_for_neg_statement(sent)

		for i, cf_stmt in enumerate(cf_stmts):
			if cf_stmt is None: continue 
			cf_tag = cf_tags[i]
			new_item = copy.copy(item)
			new_item['sent'] = cf_stmt
			new_item['orig_sent'] = item['sent']
			new_item['tag'] = cf_tag
			new_item['question_id'] = int("{0}{1:05d}".format(item['question_id'],i))
			if "yes" in item['label']:
				new_item['label'] =  {"no": 1}
			else: 
				new_item['label'] =  {"yes": 1}
			new_item['orig_label'] = item['label']
			new_item['parent_question_id'] = item['question_id']
			cf_res.append(new_item)
	
	print("Total statements ", len(cf_res))
	cf_res += data
	print(cf_counts)
	
	save_json(cfg['files']['vqa_'+split+'_si_file'], cf_res)
	# save_json("debug_vqa.json", cf_res)

'''
HELPER FUNCTIONS	
'''
def get_cfs_for_pos_statement(pos_stmt):
	cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, pos_stmt)
	exclude_idxs = [i for i,x in enumerate(cap_arr) if x in EXCLUDE_LIST]
	
	noun_chunks_2 = []
	for chunk in noun_chunks:
		if any(exc in chunk.text for exc in EXCLUDE_LIST): continue
		noun_chunks_2.append(chunk)

	cf_stmt, cf_tag = [], []
	start = time.time()
	li = get_si_comparative_antonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + len(li) * ["si_comparative_antonym"]
	append2dict(cf_counts, "si_comparative_antonym", len(li))
	# print(time.time() - start)
	start = time.time()

	li = get_si_by_noun_antonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + len(li) * ["si_noun_antonym"]
	append2dict(cf_counts, "si_noun_antonym", len(li))
	# print(time.time() - start)
	start = time.time()

	li = get_si_by_not(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + len(li) * ["si_negation"]	
	append2dict(cf_counts, "si_negation", len(li))
	# print(time.time() - start)
	start = time.time()

	li = get_si_by_subj_obj_swap(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + len(li) * ["si_subject_object_swap"]	
	append2dict(cf_counts, "si_subject_object_swap", len(li))
	# print(time.time() - start)
	start = time.time()

	li = get_si_by_num_sub(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + len(li) * ["si_number_substitution"]
	append2dict(cf_counts, "si_number_substitution", len(li))
	# print(time.time() - start)
	start = time.time()

	li = get_si_by_verb_antonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + len(li) * ["si_verb_antonym"]
	append2dict(cf_counts, "si_verb_antonym", len(li))

	# print(time.time() - start)
	start = time.time()			

	return cf_stmt, cf_tag

def get_cfs_for_neg_statement(neg_stmt):
	cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, neg_stmt)
	exclude_idxs = [i for i,x in enumerate(cap_arr) if x in EXCLUDE_LIST]
	
	noun_chunks_2 = []
	for chunk in noun_chunks:
		if any(exc in chunk.text for exc in EXCLUDE_LIST): continue
		noun_chunks_2.append(chunk)

	cf_stmt, cf_tag = [], []

	negated_li = get_si_by_not(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs)
	cf_stmt = cf_stmt + negated_li
	cf_tag = cf_tag + len(negated_li) * ["si_negation"]	
	append2dict(cf_counts, "si_negation", len(negated_li))

	for neg_stmt in negated_li:
		cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, neg_stmt)
		exclude_idxs = [cap_arr.index(exclude_item) for exclude_item in EXCLUDE_LIST if exclude_item in cap_arr ]
		
		noun_chunks_2 = []
		for chunk in noun_chunks:
			if any(exc in chunk.text for exc in EXCLUDE_LIST): continue
			noun_chunks_2.append(chunk)

		li = sp_by_comparative_synonym(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
		cf_stmt = cf_stmt + li
		cf_tag = cf_tag + ["si_comparative_antonym"] * len(li)
		append2dict(cf_counts, "si_comparative_antonym", len(li))	

		li = sp_by_number_substition(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
		cf_stmt = cf_stmt + li
		cf_tag = cf_tag + ["si_number_substitution"] * len(li)
		append2dict(cf_counts, "si_number_substitution", len(li))	

		li = sp_by_pronouns(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
		cf_stmt = cf_stmt + li
		cf_tag = cf_tag + ["si_pronoun_substitution"] * len(li)
		append2dict(cf_counts, "si_pronoun_substitution", len(li))				

	return cf_stmt, cf_tag

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

def stem(a):
	# logging.info("Stemming {}".format(a))
	doc = nlp(a)
	# logging.info("Doc {}".format(doc[0]._.lemma()))
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

def compute_and_save_embeddings():
	nouns_vocab_file = cfg['files']['vqa_nouns_vocab_file']
	with open(nouns_vocab_file, 'r') as f: 
		all_nouns = json.load(f)
	nlp = spacy.load("en_trf_bertbaseuncased_lg")
	#Get all noun embeddings
	all_noun_emb = np.array([nlp(noun).tensor for noun in tqdm(all_nouns, ascii=True)])
	all_noun_emb_dict = dict ()

	for noun in tqdm(all_nouns, ascii=True):
		all_noun_emb_dict[noun] = nlp(noun).tensor 
		
	save_pickle(cfg['files']['vqa_all_noun_emb_file'], all_noun_emb)
	save_pickle(cfg['files']['vqa_all_noun_emb_dict_file'], all_noun_emb_dict)
	print("Saved Embeddings")

def get_replacing_subject(subj, objects, all_noun_emb, all_noun_emb_dict, all_nouns, hypernyms, hyponyms):
	subj_stem = stem(subj)
	if subj in all_noun_emb_dict:
		subj_emb = all_noun_emb_dict[subj]
	elif subj_stem in all_noun_emb_dict:
		subj_emb = all_noun_emb_dict[subj_stem]
	else:
		return None
	all_dist = compute_dist(subj_emb, all_noun_emb)
	# Sorting in increasing order of distance
	# sorted_idx will contain indexes of the distance values in increasing order
	sorted_idx = np.argsort(all_dist)
	for i in sorted_idx:
		cf_subj = all_nouns[i]
		cf_subj_stem = stem(cf_subj)
		cf_subj_infl = lemminflect.getInflection(cf_subj, tag='NN')
		# Check if the key is present in the dicts
		condition_1,  condition_3, condition_l  = False, False, (Levenshtein.distance(cf_subj_stem, subj_stem) > 1)
		condition_s = subj_stem not in cf_subj_stem and cf_subj_stem not in subj_stem


		if subj in hypernyms:
			condition_1 = cf_subj not in hypernyms[subj]
		elif subj_stem in hypernyms:
			condition_1 = cf_subj not in hypernyms[subj_stem]
		
		if cf_subj in hypernyms:
			condition_3 = subj not in hypernyms[cf_subj]
		elif cf_subj_stem in hypernyms:
			condition_3 = subj not in hypernyms[cf_subj_stem]
		
		if cf_subj_stem != subj_stem and cf_subj not in objects and condition_1 and condition_3 and not (subj == 'image' and cf_subj in ['visual', 'photo', 'interface']) and not cf_subj in ['scratch', 'teenager']:
			return cf_subj

	return all_nouns[sorted_idx[-1]]

def generate_antonyms(word):
	# syn = list()
	ant = list()
	for synset in wordnet.synsets(word):
		for lemma in synset.lemmas():
			# syn.append(lemma.name())    #add the synonyms
			if lemma.antonyms():    #When antonyms are available, add them into the list
				ant.append(lemma.antonyms()[0].name())
	ants = list(ant)
	if len(ants) > 0:
		sample_ant = random.choice(ants)
		sample_ant = stem(sample_ant)
		return sample_ant

	return None

def get_replacing_number(num):
	num = num.lower()
	try:
		num_int = w2n.word_to_num(num)
		digs = int(math.log10(num_int))
	except:
		digs = 1
		num_int = 0
	increament_l= int(math.pow(10, digs))
	increament_h = int(10*math.pow(10, digs))
	new_num_int = random.choices([random.randrange(num_int + increament_l, num_int+increament_h, increament_l), 0], weights=[0.8, 0.2], k=1)[0]
	new_num = num2words(new_num_int)
	return new_num

'''
MAIN FUNCTIONS
'''
# Get cf by replacing comparatives using antonyms
def get_si_comparative_antonym(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	
	# adjectival complement or (comparative/superlative adjective)
	word_list = [i for i, x in enumerate(dep_arr) if (x == 'acomp' or (x == 'amod' and tag_arr[i] in ['JJR', 'JJS'])) and i not in exclude_idxs]
	stmt_li = []
	for sampled_idx in word_list:
		sent_arr = copy.copy(cap_arr)
		# sampled_idx = random.choice(word_list)
		comp_word = sent_arr[sampled_idx]
		ant = generate_antonyms(comp_word)
		if ant:
			sent_arr[sampled_idx] = ant
			stmt_li.append(" ".join(sent_arr))	
	return stmt_li
	
# Get CF by substituing cf subjects
def get_si_by_noun_antonym(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	
	verb_idx = [i for i, x in enumerate(pos_arr) if x == 'VERB' and i not in exclude_idxs]
	subj_idx = [i for i, x in enumerate(dep_arr) if (x in ['nsubj', 'nsubjpass'] and pos_arr[i] in ['NOUN', 'PROPN']) and i not in exclude_idxs]
	objects_idx = [i for i, x in enumerate(pos_arr) if x =='NOUN' and  i not in exclude_idxs]
	# Get unique objects and subjects
	objects = [cap_arr[i] for i in objects_idx]

	stmt_li = []
	subjects = [cap_arr[i]  for i in subj_idx]

	# Get sampled noun
	for sampled_idx in subj_idx:
		sent_arr = copy.copy(cap_arr)
		sampled_noun = sent_arr[sampled_idx]
		# Change to stem if plural
		if tag_arr[sampled_idx] in ['NNS', 'NNPS']:
			sampled_noun = stem(sampled_noun)

		# CF method using subject 
		cf_subject = get_replacing_subject(sampled_noun, objects, all_noun_emb, all_noun_emb_dict, all_nouns, hypernyms, hyponyms)
		print("sampled_noun, cf_subjec", sampled_noun, cf_subject)
		if cf_subject:
			sent_arr[sampled_idx] = cf_subject
			stmt_li.append(" ".join(sent_arr))

	for sampled_idx in objects_idx: 
		sent_arr = copy.copy(cap_arr)
		sampled_noun = sent_arr[sampled_idx]
		# Change to stem if plural
		if tag_arr[sampled_idx] in ['NNS', 'NNPS']:
			sampled_noun = stem(sampled_noun)

		# CF method using subject 
		cf_subject = get_replacing_subject(sampled_noun, objects, all_noun_emb, all_noun_emb_dict, all_nouns, hypernyms, hyponyms)
		if cf_subject:
			sent_arr[sampled_idx] = cf_subject
			stmt_li.append(" ".join(sent_arr))
	

	return stmt_li

# Generate CF using negation
def get_si_by_not(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	
	verb_idx = [i for i, x in enumerate(pos_arr) if x == 'VERB' and i not in exclude_idxs]
	subj_idx = [i for i, x in enumerate(dep_arr) if x in ['nsubj', 'nsubjpass'] and i not in exclude_idxs]
	objects_idx = [i for i, x in enumerate(pos_arr) if x =='NOUN' and i not in exclude_idxs]
	stmt_li = []
	# print(verb_idx)

	for sample_verb_idx in verb_idx:
		# Code for negation
		sent_arr = copy.copy(cap_arr)
		sample_verb = sent_arr[sample_verb_idx]
		# Past tense
		if tag_arr[sample_verb_idx] == 'VBD':
			# Get lemma for the verb
			sample_verb = stem(sample_verb)
			sent_arr[sample_verb_idx] = sample_verb
			# Add did not
			sent_arr.insert(sample_verb_idx, 'did not')
		elif tag_arr[sample_verb_idx] in ['VB', 'VBP', 'VBZ'] and len(subj_idx) > 0:
			# Singular present verb and base form
			# For Plural Noun
			if tag_arr[subj_idx[0]] in ['NNS', 'NNPS']:
				sent_arr.insert(sample_verb_idx, 'do not')
			else:
				#For Singular Noun
				sample_verb = stem(sample_verb)
				sent_arr[sample_verb_idx] = sample_verb
				sent_arr.insert(sample_verb_idx, 'does not')
		elif tag_arr[sample_verb_idx] in ['VB', 'VBG', 'VBN']:
			# For VB (Base Form), VBG(Present Participle), VBN(Past Participle)
			sent_arr.insert(sample_verb_idx, 'not')
		elif 'ADP' in pos_arr:
			sent_arr.insert(pos_arr.index('ADP'), "not")
		elif 'ADJ' in pos_arr:
			sent_arr.insert(pos_arr.index('ADJ'), "not")

		stmt_li.append(' '.join(sent_arr))


	return stmt_li

# Generate CF by subject object swapping
def get_si_by_subj_obj_swap(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	sent_arr = copy.copy(cap_arr)
	subj_idx = [i for i, x in enumerate(dep_arr) if x in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] and pos_arr[i] == 'NOUN' and i not in exclude_idxs]
	d_objects_idx = [i for i, x in enumerate(dep_arr) if x == 'dobj' and pos_arr[i] == 'NOUN' and i not in exclude_idxs ]
	p_objects_idx = [i for i, x in enumerate(dep_arr) if x == 'pobj' and pos_arr[i] == 'NOUN' and i not in exclude_idxs]

	stmt_li = []
	for sampled_subj_idx in subj_idx:

		for sampled_pobj_idx in p_objects_idx:
			sent_arr = copy.copy(cap_arr)
			# use p_objects is not empty
			if sent_arr[sampled_pobj_idx] != sent_arr[sampled_subj_idx]:
				sent_arr[sampled_pobj_idx], sent_arr[sampled_subj_idx] = sent_arr[sampled_subj_idx], sent_arr[sampled_pobj_idx]
				stmt_li.append(" ".join(sent_arr))

		for sampled_dobj_idx in d_objects_idx:
			sent_arr = copy.copy(cap_arr)
			# Use d_objects if not empty
			if sent_arr[sampled_dobj_idx] !=  sent_arr[sampled_subj_idx]:
				sent_arr[sampled_dobj_idx], sent_arr[sampled_subj_idx] = sent_arr[sampled_subj_idx], sent_arr[sampled_dobj_idx]
				stmt_li.append(" ".join(sent_arr))

	return stmt_li
	
# Generate cfs by changing numerical entities
def get_si_by_num_sub(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	
	num_idxs = [i for i, x in enumerate(dep_arr) if x == 'nummod' and i not in exclude_idxs]
	stmt_li = []
	for sample_num_idx in num_idxs:
		sent_arr = copy.copy(cap_arr)
		cf_num = get_replacing_number(sent_arr[sample_num_idx])
		sent_arr[sample_num_idx] = cf_num
		stmt_li.append(" ".join(sent_arr))

	return stmt_li
	
# Generate cf by verb antonym replacement
def get_si_by_verb_antonym(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):

	
	verb_idx = [i for i, x in enumerate(pos_arr) if x == 'VERB' and i not in exclude_idxs]
	stmt_li = []
	for sampled_verb_idx in verb_idx:
		sent_arr = copy.copy(cap_arr)	
		sampled_verb = cap_arr[sampled_verb_idx]
		ant_verb = generate_antonyms(sampled_verb)
		
		if ant_verb:
			inflect_ant_verb = lemminflect.getInflection(ant_verb,tag=tag_arr[sampled_verb_idx])
			try:
				sent_arr[sampled_verb_idx] = inflect_ant_verb[0]
			except: 
				sent_arr[sampled_verb_idx] = ant_verb
			stmt_li.append(" ".join(sent_arr))
	return stmt_li

'''
TEST FUNCTIONS
'''	
def test_si_by_comparative_antonym():
	sentences = ["This is a smaller man", "Jupiter is the biggest planet in our solar system.", "The girl is shorter than boy", "I got higher marks", "the ground is big"]
	for q in sentences:
		q_ant = get_si_comparative_antonym(q)
		print(q, ",",  q_ant)

def test_si_by_noun_antonym():
	sentences = ["An image shows just an adult gorilla glaring forward and on all fours"]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, sent)
		print("Cap_Arr", cap_arr)
		print("dep_arr", dep_arr)
		print("tag_arr", tag_arr)
		print("pos_arr", pos_arr)
		print("Noun_chunks", noun_chunks)
		# q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs
		print(sent, get_si_by_noun_antonym(sent, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, []))
		
def test_si_by_not():
	res = []
	count = 0
	sentences = ["The boy is reading the book", "The man watched football", "The girl is sleeping", "The lady is watching the game and drinking coffee", "the boys play cricket", "the boy plays cricket","the boy is playing cricket", "the boy played cricket", "the boy had played cricket", "the boy has been playing cricket"]
	for q in sentences:
		cf_stmt = get_si_by_not(q)
		print(q, ",", cf_stmt)

def test_si_by_subj_obj_swap():
	res = []
	count = 0
	sentences = ["each image contains a pair not of hand coverings , and one image contains a pair of gloves with five full - length fingers that cover the finger tips .",
				"the left image features a single fur - trimmed fingerless mitten not with small embellishments dotting its front , and the right image shows a pair of fur - trimmed half - mitts with no thumb part showing .", 
				"one image contains one man dressed in a beer bottle costume , and the other image shows a row of at least three people who wear similar beer costumes ."]
	for q in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, q)
		exclude_idxs = [i for i,x in enumerate(cap_arr) if x in EXCLUDE_LIST]
		noun_chunks_2 = []
		for chunk in noun_chunks:
			if any(exc in chunk.text for exc in EXCLUDE_LIST): continue
			noun_chunks_2.append(chunk)

		print(exclude_idxs)
		print(noun_chunks_2)
		cf_stmt = get_si_by_subj_obj_swap(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
		print(q, ",", cf_stmt)

def test_si_by_num_sub():
	res = []
	count = 0
	sentences = ["There are three dogs in the image", "There are zero dogs in the image", "The car is a four seater", "The truck can handle 4000 kgs", "the movie costs 100 rupees"]
	for q in sentences:
		cf_stmt = get_si_by_num_sub(q)
		print(q, ",", cf_stmt)	

def test_si_by_verb_antonym():
	res = []
	count = 0
	sentences = ["the man is cleaning the clothes", "the man is shouting in the streets", "The girl is sleeping on the bed", "the boy is riding a bike", "the man is singing a song", "the man is watching the tv", "the boy is flying a kite"]
	for q in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr = get_tags(q)
		cf_stmt = get_si_by_verb_antonym(q, cap_arr, dep_arr, tag_arr, pos_arr)
		print(q, ",", cf_stmt)	


nouns_vocab_file = cfg['files']['nlvr_nouns_vocab_file']
hypernyms_file = cfg['files']['nlvr_hypernyms_file']
hyponyms_file = cfg['files']['nlvr_hyponyms_file']

all_noun_emb = load_pickle(cfg['files']['nlvr_all_noun_emb_file'])
all_noun_emb_dict = load_pickle(cfg['files']['nlvr_all_noun_emb_dict_file'])

with open(nouns_vocab_file, 'r') as f: 
	all_nouns = json.load(f)

with open(hypernyms_file, 'r') as f: 
	hypernyms = json.load(f)	

with open(hyponyms_file, 'r') as f: 
	hyponyms = json.load(f)

if __name__ == '__main__':
	format = "%(asctime)s: %(message)s"
	logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
	logging.getLogger().setLevel(logging.DEBUG)
	test_si_by_noun_antonym()
