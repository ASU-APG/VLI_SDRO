from nltk.corpus import wordnet
from tqdm import tqdm 
import logging
import json
import yaml
file = open('config.yaml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

def get_hyponyms(word):
	syn_arr = wordnet.synsets(word)
	all_hypo = []
	for syn in syn_arr:
		hypo_syn_arr = syn.hyponyms()
		for hypo_syn in hypo_syn_arr:
			all_hypo += hypo_syn.lemma_names()

	all_hypo = list(set(all_hypo))
	return all_hypo

def get_hypernyms(word):
	syn_arr = wordnet.synsets(word)
	all_hyper = []
	for syn in syn_arr:
		hyper_syn_arr = syn.hypernyms()
		for hyper_syn in hyper_syn_arr:
			all_hyper += hyper_syn.lemma_names()

	all_hyper = list(set(all_hyper))
	return all_hyper

def main():
	format = "%(asctime)s: %(message)s"
	logging.basicConfig(format=format, level=logging.INFO,
						datefmt="%H:%M:%S")
	nouns_vocab_file = cfg['files']['vqa_nouns_vocab_file']
	hypernym_file = cfg['files']['vqa_hypernyms_file']
	hyponym_file = cfg['files']['vqa_hyponyms_file']

	hypernyms = dict()
	hyponyms = dict()

	with open(nouns_vocab_file, 'r') as f: 
		all_nouns = json.load(f)

	for word in tqdm(all_nouns):
		hypernyms[word] = get_hypernyms(word)
		hyponyms[word] = get_hyponyms(word)
	logging.info("Done")
	with open(hypernym_file, 'w') as fp:
		json.dump(hypernyms, fp, sort_keys=True, indent=4)

	with open(hyponym_file, 'w') as fp:
		json.dump(hyponyms, fp, sort_keys=True, indent=4)

if __name__ == '__main__':
	# word = "man"
	# print(get_hyponyms(word))
	# print(get_hypernyms(word))
	main()