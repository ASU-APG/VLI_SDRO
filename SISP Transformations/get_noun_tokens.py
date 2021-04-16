import json
# import pytorch_transformers
import nltk
import logging
import spacy
import lemminflect
import yaml
from tqdm import tqdm
file = open('config.yaml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

nltk.download('punkt')
nlp = spacy.load('en_core_web_lg')

def stem(a):
    # logging.info("Stemming {}".format(a))
    doc = nlp(a)
    # logging.info("Doc {}".format(doc[0]._.lemma()))
    a = doc[0]._.lemma()
    return a

def tokenize(caption, tokenizer):
    token_ids = tokenizer.encode(caption,add_special_tokens=True)
    tokens = [
        tokenizer._convert_id_to_token(t_id) for t_id in token_ids]
    return token_ids, tokens

def combine_subtokens(tokens,i):
    ## assumes subtoken starts at i
    count = 1
    word = tokens[i]
    for j in range(i+1,len(tokens)):
        token = tokens[j]
        if len(token)>2 and token[:2]=='##':
            count += 1 
            word += token[2:]
        else:
            break
    
    return word, count

def align_pos_tokens(pos_tags,tokens):
    alignment = [None]*len(pos_tags)
    for i in range(len(alignment)):
        alignment[i] = []
    
    token_len = len(tokens)
    last_match = -1
    skip_until = -1
    for i, (word,tag) in enumerate(pos_tags):
        for j in range(last_match+1,token_len):
            if j < skip_until:
                continue
            
            if j==skip_until:
                skip_until = -1

            token = tokens[j]
            if word==token:
                alignment[i].append(j)
                last_match = j
                break
            elif len(token)>2 and token[:2]=='##':
                combined_token, sub_token_count = combine_subtokens(tokens,j-1)
                skip_until = j-1+sub_token_count
                if word==combined_token:
                    for k in range(sub_token_count):
                        alignment[i].append(k+j-1)
                        last_match = j-1+sub_token_count-1
                 
    return alignment

def get_noun_token_ids(pos_tags,alignment=None):
    noun_words = set()
    verbs = set()
    import sys
    token_ids = []
    for i, (word,tag) in enumerate(pos_tags):
        if tag in ['NN','NNS','NNP','NNPS']:
            # Get lemma for plural nouns
            if tag in ['NNS', 'NNPS']:
                word = stem(word)
            noun_words.add(word)
            # for idx in alignment[i]:
            #     token_ids.append(idx)

    return token_ids, noun_words

def group_token_ids(token_ids,tokens):
    grouped_ids = []
    group_num = -1
    for token_id in token_ids:
        token = tokens[token_id]
        if len(token)>=2 and token[:2]=='##':
            grouped_ids[-1].append(token_id)
        else:
            grouped_ids.append([token_id])
    
    return grouped_ids

def ignore_words_from_pos(pos_tags,words_to_ignore):
    for i in range(len(pos_tags)):
        word, tag = pos_tags[i]
        if word in words_to_ignore:
            pos_tags[i] = (word,'IG')
        
    return pos_tags
def main_helper(statement, tokenizer):
    # token_ids, tokens = tokenize(statement, tokenizer)
    nltk_tokens = nltk.word_tokenize(statement.lower())
    pos_tags = nltk.pos_tag(nltk_tokens)
    pos_tags = ignore_words_from_pos(pos_tags,['is','has','have','had','be'])
    noun_token_ids_, noun_words = get_noun_token_ids(pos_tags)
    ret_dict = {"words" : list(noun_words)}
    return ret_dict, noun_words

def main_violin():
    annotations_file = cfg['violin_annotations']
    noun_vocab_json_file = cfg['nouns_vocab_file']
    data = json.load(open(annotations_file))
    tokenizer = None
    
    noun_vocab = set()
    count = 0
    for clip_id, data_item in data.items():
        count += 1
        logging.info("####### INDEX :: {} PROCESSING :: {} ########".format(count, clip_id))
        item_dict = dict()
        item_dict['statements'] = []
        item_dict['subtitles'] = []
        item_dict['clip_id'] = clip_id
        item_statements = data_item['statement']
        for statement_pair in item_statements:
            # +ve statements
            ret_dict_1, noun_words_1 = main_helper(statement_pair[0], tokenizer)
            # -ve statements
            ret_dict_2, noun_words_2 = main_helper(statement_pair[1], tokenizer)
            if "mans" in noun_words_1 or "mans" in noun_words_2:
                print(statement_pair)
            noun_vocab.update(noun_words_1)
            noun_vocab.update(noun_words_2)
    
    noun_vocab = list(noun_vocab)

    with open(noun_vocab_json_file, 'w') as fp:
        json.dump(noun_vocab, fp, sort_keys=True, indent=4)

def main_nlvr():
    annotations_file = cfg['files']['nlvr_train_file']
    noun_vocab_json_file = cfg['files']['nlvr_nouns_vocab_file']
    data = json.load(open(annotations_file))
    tokenizer = None
    
    noun_vocab = set()
    count = 0
    for item in tqdm(data):
        sent = item['sent']
        ret_dict_1, noun_words_1 = main_helper(sent, tokenizer)
        noun_vocab.update(noun_words_1)    
    noun_vocab = list(noun_vocab)

    with open(noun_vocab_json_file, 'w') as fp:
        json.dump(noun_vocab, fp, sort_keys=True, indent=4)

def main_vqa():
    annotations_file = cfg['files']['vqa_train_file']
    noun_vocab_json_file = cfg['files']['vqa_nouns_vocab_file']
    data = json.load(open(annotations_file))
    tokenizer = None
    
    noun_vocab = set()
    count = 0
    for item in tqdm(data):
        sent = item['sent']
        ret_dict_1, noun_words_1 = main_helper(sent, tokenizer)
        noun_vocab.update(noun_words_1)    
    noun_vocab = list(noun_vocab)

    with open(noun_vocab_json_file, 'w') as fp:
        json.dump(noun_vocab, fp, sort_keys=True, indent=4)

if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.DEBUG)
    main_vqa()
    
logging.info("Done")
