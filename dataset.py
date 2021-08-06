## Import libraries ##
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm
import math, random
## PyTorch ##
import torch
from torch.utils.data import DataLoader, Dataset


def generate_indices(target_len, total_len = 50):
	
	i = target_len - 1
	
	average = (total_len - 1) / i
	a = math.floor(average)
	b = math.ceil(average)
	
	for j in range(1, i + 1):
		if j * a + (i - j) * b == total_len - 1:
			break
	gaps = [a] * j + [b] * (i - j)
	random.shuffle(gaps)
	k = 0
	ans = [0]
	for val in gaps:
		ans.append(ans[-1] + val)
	return torch.tensor(ans)

## Define helpful functions ##
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

## Define a vocabulary class ##
class Vocabulary:
	def __init__(self, freq_threshold):
		self.itos = { 0:'<EOS>', 1:'<UNK>'}
		self.stoi = {'<EOS>': 0, '<UNK>':1}
		self.freq_threshold = freq_threshold
		self.freq_dict = defaultdict(int)
		
	def __len__(self):
		return len(self.itos)
	
	def buildVocab(self, sentence_list):
		idx = len(self.itos)
		for i, sent in enumerate(tqdm(sentence_list)):
			for word in sent:
				self.freq_dict[word] += 1
				
				if self.freq_dict[word] == self.freq_threshold:
					self.stoi[word] = idx
					self.itos[idx] = word
					idx += 1

		
	def numer(self, sent):
		return [self.stoi[word] if word in self.stoi else self.stoi['<UNK>'] for word in sent] + [self.stoi['<EOS>']]

class ds(Dataset):
	def __init__(self, text_data):
		self.text_data = text_data
	def __len__(self):
		return len(self.text_data)
	def __getitem__(self, index):
		sentence = self.text_data[index]
		return torch.tensor(sentence)

def get_loaders():
	# load the vocabulary from "eng-fra.txt"
	lines = open("./eng-fra.txt").read().strip().split('\n')
	# normalize the text
	full_text_data = [[normalizeString(s) for s in l.split('\t')][0] for l in lines]
	# split sentences into words
	full_text_data = [sentence.split() for sentence in full_text_data]
	print(f"Total number of sentences is {len(full_text_data)} \n")
	fullVocab = Vocabulary(5)
	fullVocab.buildVocab(full_text_data)
	print(f"The vocabulary size if {len(fullVocab)} \n")
	
	min_len, max_len = 5, 14
	new_text = [sent for sent in full_text_data if min_len <= len(sent) <= max_len]
	print(f"The size of the newly created text data is {len(new_text)}, {(len(new_text)*100)//len(full_text_data)}% of the original text")
	# perSentence dictionary maps length of the sentence to all possible sentences of that length
	perSentence = defaultdict(list)
	for sent in new_text:
		num_sentence = fullVocab.numer(sent)
		perSentence[len(sent)].append(num_sentence)
	## assertion check
	assert (sum([len(perSentence[d]) for d in perSentence]) == len(new_text))

	return perSentence, fullVocab


if __name__ == "__main__":
	get_loaders()

