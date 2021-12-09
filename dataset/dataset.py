## Import libraries ##

from collections import defaultdict
from tqdm import tqdm
import os

from utils import *
## PyTorch ##
import torch
from torch.utils.data import DataLoader, Dataset


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

class custom_dataset(Dataset):
    def __init__(self, text_data):
        self.text_data = text_data
    def __len__(self):
        return len(self.text_data)
    def __getitem__(self, index):
        sentence = self.text_data[index]
        return torch.tensor(sentence)

def get_loaders(path, freq_threshold = 5, min_len = 5, max_len = 14):
    # load the vocabulary from "eng-fra.txt"
    lines = open(path).read().strip().split('\n')

    # normalize the text
    full_text_data = [[normalizeString(s) for s in l.split('\t')][0] for l in lines]

    # split sentences into words
    full_text_data = [sentence.split() for sentence in full_text_data]
    print(f"Total number of sentences is {len(full_text_data)} \n")

    fullVocab = Vocabulary(freq_threshold)
    fullVocab.buildVocab(full_text_data)
    print(f"The vocabulary size if {len(fullVocab)} \n")
    
    # filter sentences by length
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
    path = os.path.join("../data", "eng-fra.txt")
    get_loaders(path)

