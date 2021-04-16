import random 
import spacy 
import json 
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
from FairSeqNmt.nmt import MY_NMT
from utils import get_tags, get_num, get_closest_word, get_opposite_word, get_double_negative_for_list, get_synset_definition, save_json
from pass2act import pass2act

EXCLUDE_LIST = ["photo", "photos", "photograph", "photographs", 
				"picture", "pictures", "image", "images", "show", "shows", "feature", 
				"features", "contain", "contains", "depict", "depicts", "display", "displays", 
				"visible", "displayed", "depicted", "appear", "appears", "none", "right", "left", "interface", "interfaces"]

file = open('config.yaml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
nlp = spacy.load('en_core_web_lg')
nmt = MY_NMT()

def append2dict(d, key, val):
	if key in d:
		d[key] += val 
	else: 
		d[key] = val 


cf_counts = dict()
def get_violin_sp_sentences():
	annotations_file = cfg['files']['violin_annotations']
	res = dict()

	with open(annotations_file, 'r') as f: 
		all_data = json.load(f)

	data = []
	for clip_id, data_item in tqdm(all_data.items()):
		data.append(data_item)

	for data_item in tqdm(data):
		clip_id = data_item['file']
		item_statements = data_item['statement']
		tags_li, stmts_li = [], []
		for statement_pair in item_statements:
			# Using the negative statement to get the sp
			pos_stmt, neg_stmt = statement_pair[0], statement_pair[1]

			sp_stmts1, sp_tags1 = get_sp_sentences_for_pos_statement(pos_stmt)
			sp_stmts2, sp_tags2 = get_sp_sentences_for_neg_statement(neg_stmt)

			tags = [sp_tags1, sp_tags2]
			stmts = [sp_stmts1, sp_stmts2]

			tags_li.append(tags)
			stmts_li.append(stmts)

		data_item['sp_statements'] = stmts_li
		data_item['sp_tags'] = tags_li
		res[clip_id] = data_item
	# print(cf_counts)
	print("Category-wise count ", cf_counts)
	# print("Total statements ", len(list(res.keys())))
	save_json(cfg['files']['violin_sp_annotations'], res)
	# save_json("violin_debug_sp.json", res)

def get_nlvr2_sp_sentences(split):
	file_path = cfg['files']['nlvr_'+split+'_si_file']
	# file_path = cfg['files']['nlvr_'+split+'_file']
	# file_path = "debug_si.json"
	with open(file_path, 'r') as f: 
		data = json.load(f)
	cf_res = list()
	data_pos_li, data_neg_li = [], []
	# for item in data:
	# # 	# Only taking the +ve statements
	# 	item['tag'] = 'orig'
	# 	if item['label'] == 1:
	# 		data_pos_li.append(item)
	# 	else: 
	# 		data_neg_li.append(item)
	
	# count_dict = dict()
	uid_counter = len(data) + 1
	for item in tqdm(data):
		if item['tag'] !=  'orig': continue
		img0_id = item['img0'].replace("-img0", "")
		sent = item['sent']
		if item['label']:
			cf_stmts, cf_tags = get_sp_sentences_for_pos_statement(sent)
		else: 
			cf_stmts, cf_tags = get_sp_sentences_for_neg_statement(sent)

		for i, cf_stmt in enumerate(cf_stmts):
			if cf_stmt is None: continue 
			cf_tag = cf_tags[i]
			new_item = copy.copy(item)
			new_item['sent'] = cf_stmt
			new_item['orig_sent'] = item['sent']
			new_item['tag'] = cf_tag
			new_item['uid'] = "nlvr2_"+split+"_{}".format(uid_counter)
			uid_counter += 1

			new_item['label'] = item['label']
			new_item['orig_label'] = item['label']

			# new_item['label'] = 0
			new_item['parent_identifier'] = item['identifier']
			new_item['parent_uid'] = item['uid']
			new_item['identifier'] = "{}-{}".format(img0_id, uid_counter)
			cf_res.append(new_item)
		# print(cf_counts)

	print("counts", cf_counts)
	print("Total statements ", len(cf_res))
	cf_res += data
	save_json(cfg['files']['nlvr_'+split+'_si_sp_file'], cf_res)
	# save_json("debug_si_sp.json", cf_res)

def get_vqa_sp_sentences(split):
	print("Inside vqa")
	# file_path = cfg['files']['vqa_'+split+'_si_file']
	file_path = cfg['files']['vqa_'+split+'_file']
	with open(file_path, 'r') as f: 
		data = json.load(f)
	cf_res = list()
	data_li = []
	for item in data:
		# if item['label']:
		item['tag'] = 'orig'
		data_li.append(item)
	# for item in data:
	# 	# Only taking the +ve statements
	# 	# if item['label'] == 1:
	# 	if 'tag' not in item:
	# 		data_li.append(item)
	
	# count_dict = dict()
	uid_counter = len(data) + 1
	for item in tqdm(data_li):
		if item['tag'] != 'orig' : continue
		img0_id = item['img_id']
		sent = item['sent']
		if 'yes' in item['label']:
			cf_stmts, cf_tags = get_sp_sentences_for_pos_statement(sent)
		else: 
			cf_stmts, cf_tags = get_sp_sentences_for_neg_statement(sent)
		for i, cf_stmt in enumerate(cf_stmts):
			if cf_stmt is None: continue 
			cf_tag = cf_tags[i]
			new_item = copy.copy(item)
			new_item['sent'] = cf_stmt
			new_item['orig_sent'] = item['sent']
			new_item['tag'] = cf_tag
			new_item['question_id'] = int("{0}{1:05d}".format(item['question_id'],13+i))
			if "no" in item['label']:
				new_item['label'] =  {"no": 1}
			else: 
				new_item['label'] =  {"yes": 1}
			new_item['orig_label'] = item['label']
			new_item['parent_question_id'] = item['question_id']
			cf_res.append(new_item)
		# print(cf_counts)
	# cf_res += data
	print(cf_counts)
	print("Total statements ", len(cf_res))
	# save_json("debug_vqa_sp.json", cf_res)
	save_json(cfg['files']['vqa_'+split+'_sp_file'], cf_res)

def get_sp_sentences_for_pos_statement(pos_stmt):
	cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, pos_stmt)
	exclude_idxs = [i for i,x in enumerate(cap_arr) if x in EXCLUDE_LIST]
	
	noun_chunks_2 = []
	for chunk in noun_chunks:
		if any(exc in chunk.text for exc in EXCLUDE_LIST): continue
		noun_chunks_2.append(chunk)
	cf_stmt, cf_tag = [], []
	start = time.time()

	li = sp_by_pronouns(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	# print(pos_stmt, li)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_pronoun_substitution"] * len(li)
	append2dict(cf_counts, "sp_pronoun_substitution", len(li))
	
	# print(time.time() - start)
	# start = time.time()
	li = sp_by_number_substition(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_number_substitution"] * len(li)
	append2dict(cf_counts, "sp_number_substitution", len(li))


	# # print(time.time() - start)
	# start = time.time()
	li = sp_by_comparative_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_comparative_synonym"] * len(li)
	append2dict(cf_counts, "sp_comparative_synonym", len(li))


	# cf_stmt.append(sp_by_comparative_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr))
	# cf_tag.append("sp_comparative_synonym")
	# # print(time.time() - start)
	# start = time.time()

	li = sp_by_verb_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_verb_synonym"] * len(li)
	append2dict(cf_counts, "sp_verb_synonym", len(li))


	# cf_stmt.append(sp_by_verb_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr))
	# cf_tag.append("sp_verb_synonym")
	# # print(time.time() - start)
	# start = time.time()

	li = sp_noun_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_noun_synonym"] * len(li)
	append2dict(cf_counts, "sp_noun_synonym", len(li))


	# cf_stmt.append(sp_noun_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr))
	# cf_tag.append("sp_noun_synonym")
	# # print(time.time() - start)
	# start = time.time()
	
	cf_stmt.append(sp_by_nmt(pos_stmt))
	cf_tag.append("sp_nmt")
	append2dict(cf_counts, "sp_nmt", 1)

	# print(time.time() - start)
	# start = time.time()	
	# for i, cf in enumerate(cf_stmt):

	# 	if cf_tag[i] not in cf_counts:
	# 		cf_counts[cf_tag[i]] = 0

	# 	cf_counts[cf_tag[i]] += 1	
	return cf_stmt, cf_tag

# to generate sps for negative statement
def get_sp_sentences_for_neg_statement(neg_stmt):
	cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, neg_stmt)
	exclude_idxs = [i for i,x in enumerate(cap_arr) if x in EXCLUDE_LIST]

	noun_chunks_2 = []
	for chunk in noun_chunks:
		if any(exc in chunk.text for exc in EXCLUDE_LIST): continue
		noun_chunks_2.append(chunk)
	cf_stmt, cf_tag = [], []
	start = time.time()

	li = sp_by_pronouns(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	# print(pos_stmt, li)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_pronoun_substitution"] * len(li)
	append2dict(cf_counts, "sp_pronoun_substitution", len(li))
	
	# print(time.time() - start)
	# start = time.time()
	li = sp_by_number_substition(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs, exactly=True)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_number_substitution"] * len(li)
	append2dict(cf_counts, "sp_number_substitution", len(li))


	# # print(time.time() - start)
	# start = time.time()
	li = sp_by_comparative_synonym(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_comparative_synonym"] * len(li)
	append2dict(cf_counts, "sp_comparative_synonym", len(li))


	# cf_stmt.append(sp_by_comparative_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr))
	# cf_tag.append("sp_comparative_synonym")
	# # print(time.time() - start)
	# start = time.time()

	li = sp_by_verb_synonym(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_verb_synonym"] * len(li)
	append2dict(cf_counts, "sp_verb_synonym", len(li))


	# cf_stmt.append(sp_by_verb_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr))
	# cf_tag.append("sp_verb_synonym")
	# # print(time.time() - start)
	# start = time.time()

	li = sp_noun_synonym(neg_stmt, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks_2, exclude_idxs)
	cf_stmt = cf_stmt + li
	cf_tag = cf_tag + ["sp_noun_synonym"] * len(li)
	append2dict(cf_counts, "sp_noun_synonym", len(li))


	# cf_stmt.append(sp_noun_synonym(pos_stmt, cap_arr, dep_arr, tag_arr, pos_arr))
	# cf_tag.append("sp_noun_synonym")
	# # print(time.time() - start)
	# start = time.time()
	
	cf_stmt.append(sp_by_nmt(neg_stmt))
	cf_tag.append("sp_nmt")
	append2dict(cf_counts, "sp_nmt", 1)

	# print(time.time() - start)
	# start = time.time()	
	# for i, cf in enumerate(cf_stmt):

	# 	if cf_tag[i] not in cf_counts:
	# 		cf_counts[cf_tag[i]] = 0

	# 	cf_counts[cf_tag[i]] += 1	
	return cf_stmt, cf_tag


'''
MAIN FUNCTIONS
'''
def sp_by_number_substition(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs, exactly = False):
	'''
	sp_by_number_substition is used to replace numbers in the sentence with a randomly choosen equal, lesser or greater than
	number representation
	'''
	
	num_idxs = [i for i, x in enumerate(dep_arr) if x == 'nummod' and i not in exclude_idxs]

	stmt_li = []
	for sample_num_idx in num_idxs:
		sent_arr = copy.copy(cap_arr)
		if (sample_num_idx > 0 and sent_arr[sample_num_idx-1] == 'exactly') or (sample_num_idx < len(num_idxs)-1 and sent_arr[sample_num_idx+1] == 'image'):
			cf_num = get_num(sent_arr[sample_num_idx], exactly = True)
		else: 
			cf_num = get_num(sent_arr[sample_num_idx], exactly = exactly)
		sent_arr[sample_num_idx] = cf_num
		stmt_li.append(" ".join(sent_arr))

	return stmt_li    

def sp_by_comparative_synonym(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	'''
	sp_by_comparative_synonym is used to replace a comparative with a synonym
	'''
	word_list = [i for i, x in enumerate(dep_arr) if (x == 'acomp' or (x == 'amod' and tag_arr[i] in ['JJR', 'JJS'])) and i not in exclude_idxs]
	stmt_li = []
	for sampled_idx in word_list:
		sent_arr = copy.copy(cap_arr)
		comp_word = sent_arr[sampled_idx]
		synonyms = get_closest_word(comp_word)
		for synonym in synonyms:
			sent_arr[sampled_idx] = synonym
			stmt_li.append(" ".join(sent_arr))
	return stmt_li

def sp_by_verb_synonym(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	'''
	sp_by_verb_synonym is used to replace a verb by synonym
	'''
	
	verb_idx = [i for i, x in enumerate(pos_arr) if x == 'VERB' and i not in exclude_idxs]
	stmt_li = []
	for sampled_verb_idx in verb_idx:
		sent_arr = copy.copy(cap_arr)
		sampled_verb = sent_arr[sampled_verb_idx]
		verb_synonyms = get_closest_word(sampled_verb, ret_count=2)

		for verb_synonym in verb_synonyms:
			sent_arr[sampled_verb_idx] = verb_synonym
			stmt_li.append(" ".join(sent_arr))

	return stmt_li

def sp_noun_synonym(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	'''
	sp_noun_synonym is used to get synonyms for subjects and objects
	'''
	
	subj_idx = [i for i, x in enumerate(dep_arr) if x == 'nsubj' and pos_arr[i] == 'NOUN' and i not in exclude_idxs]
	d_objects_idx = [i for i, x in enumerate(dep_arr) if x == 'dobj' and pos_arr[i] == 'NOUN' and i not in exclude_idxs]
	p_objects_idx = [i for i, x in enumerate(dep_arr) if x == 'pobj' and pos_arr[i] == 'NOUN' and i not in exclude_idxs]
	flag = False
	li = []

	if len(subj_idx) > 0:
		for sampled_subj_idx in subj_idx:
			sent_arr = copy.copy(cap_arr)
			sampled_subject = sent_arr[sampled_subj_idx]
			synonyms = get_closest_word(sampled_subject)
			for synonym in synonyms:
				sent_arr[sampled_subj_idx] = synonym
				subj_syn_sent = " ".join(sent_arr)
				li.append(subj_syn_sent)

	if len(p_objects_idx) > 0 :
		# use p_objects if not empty
		for sampled_pobj_idx in p_objects_idx:
			sent_arr = copy.copy(cap_arr)
			sampled_pobj = sent_arr[sampled_pobj_idx]
			synonyms = get_closest_word(sampled_pobj)
			for synonym in synonyms:
				sent_arr[sampled_pobj_idx] = synonym
				pobj_syn_sent =  " ".join(sent_arr)
				li.append(pobj_syn_sent)

	if len(d_objects_idx) > 0:
		# Use d_objects if not empty
		for sampled_dobj_idx in d_objects_idx:
			sent_arr = copy.copy(cap_arr)
			sampled_dobj = sent_arr[sampled_dobj_idx]
			synonyms = get_closest_word(sampled_dobj)
			for synonym in synonyms:
				sent_arr[sampled_dobj_idx] = synonym
				dobj_syn_sent = " ".join(sent_arr)
				li.append(dobj_syn_sent)
	return li

def sp_by_pronouns(q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs):
	subj_idx = [i for i, x in enumerate(dep_arr) if x == 'nsubj' and pos_arr[i] in ['NOUN'] and i not in exclude_idxs]
	stmt_li = []
	for sampled_idx in subj_idx:
		sampled_subject = cap_arr[sampled_idx]
		for chunk in noun_chunks:
			sent_arr = copy.copy(cap_arr)
			if sampled_idx >= chunk.root.i - len(chunk) + 1 and sampled_idx <= chunk.root.i:
				chunk_idx = chunk.root.i
				chunk_len = len(chunk)
				# Replacing plurals
				if sent_arr[chunk_idx] in ['man', 'boy', 'guy', 'lord', 'husband', 'father', 'boyfriend', 'son', 'brother', 'grandfather', 'uncle']:
					for idx in range(chunk_idx-chunk_len+1, chunk_idx):
						sent_arr[idx] = ''
					sent_arr[chunk_idx] = random.choices(['he', 'someone', 'somebody'], weights=[0.6, 0.2, 0.2], k=1)[0]
					
				elif sent_arr[chunk_idx] in ['woman', 'girl', 'lady', 'wife', 'mother', 'daughter', 'sister', 'girlfriend', 'grandmother', 'aunt']:
					for idx in range(chunk_idx-chunk_len+1, chunk_idx):
						sent_arr[idx] = ''
					sent_arr[chunk_idx] = random.choices(['she', 'someone', 'somebody'], weights=[0.6, 0.2, 0.2], k=1)[0]

				elif tag_arr[sampled_idx] in ['NN', 'NNP']:
					for idx in range(chunk_idx-chunk_len+1, chunk_idx):
						sent_arr[idx] = ''
					sent_arr[sampled_idx] = random.choice(['something'])

				else:
					for idx in range(chunk_idx-chunk_len+1, chunk_idx):
						sent_arr[idx] = ''
					sent_arr[sampled_idx] = 'they' 
				
				stmt_li.append(" ".join(sent_arr))
				break
	return stmt_li

def sp_by_nmt(q):
	'''
	sp_by_nmt uses fairseq NMT to generate english-english translation of input q
	'''
	return nmt.translate(q)

'''
TEST FUNCTIONS
'''	

def test_sp_noun_synonym():
	# sentences = ["The lady in the black tanktop looked out the window to make sure it was safe to open it.","The boy is reading the book", "The man is watching football", "The girl is sleeping on the bed", "The lady is watching the game and drinking coffee"]
	sentences = ["A paved road passes near the house in the image on the right"]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, sent)
		print("Cap_Arr", cap_arr)
		print("dep_arr", dep_arr)
		print("tag_arr", tag_arr)
		print("pos_arr", pos_arr)
		print("Noun_chunks", noun_chunks)
		# q, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, exclude_idxs
		print(sent, sp_noun_synonym(sent, cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks, []))

def test_sp_by_verb_synonym():
	sentences = ["the man is cleaning the clothes", "the man is shouting in the streets", "The girl is sleeping on the bed", "the boy is riding a bike", "the man is singing a song", "the man is watching the tv", "the boy is flying a kite"]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr = get_tags(nlp, sent)
		print(sent, sp_by_verb_synonym(sent, cap_arr, dep_arr, tag_arr, pos_arr))

def test_sp_by_number_substition():
	sentences = ["There are three dogs in the image", "There are zero dogs in the image", "The car is a four seater", "The truck can handle 4000 kgs", "the movie costs 100 rupees"]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr = get_tags(nlp, sent)
		print(sent, sp_by_number_substition(sent, cap_arr, dep_arr, tag_arr, pos_arr))

def test_sp_by_comparative_synonym():
	sentences = ["This is a smaller man", "Jupiter is the biggest planet in our solar system.", "The girl is shorter than boy", "I got higher marks", "the ground is big"]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr = get_tags(nlp, sent)
		print(sent, sp_by_comparative_synonym(sent, cap_arr, dep_arr, tag_arr, pos_arr))

def test_sp_double_negative():
	sentences = ["The lady in the black tanktop looked out the window to make sure it was safe to open it.", "The man in the blue shirt believes that this is not a good time to talk ", "the girl is eating a cookie", "the boys play basketball"]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr = get_tags(nlp, sent)
		print(sent, sp_double_negative(sent, cap_arr, dep_arr, tag_arr, pos_arr))

def test_sp_by_pronouns():
	# sentences = ["The lady in the black tanktop looked out the window to make sure it was safe to open it.","The boy is reading the book", "The man is watching football", "The girl is sleeping on the bed", "The lady is watching the game and drinking coffee"]
	sentences = ["The lady in the black tanktop looked out the window to make sure it was safe to open it."]
	for sent in sentences:
		cap_arr, dep_arr, tag_arr, pos_arr, noun_chunks = get_tags(nlp, sent)
		print(noun_chunks)
		print(sent, sp_by_pronouns(sent, cap_arr, dep_arr, tag_arr, pos_arr))	

if __name__ == '__main__':
	format = "%(asctime)s: %(message)s"
	logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
	logging.getLogger().setLevel(logging.DEBUG)
	test_sp_noun_synonym()








	
