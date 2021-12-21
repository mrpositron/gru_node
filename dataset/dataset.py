## Import libraries ##

from collections import defaultdict
from tqdm import tqdm
import os

from .utils import *
## PyTorch ##
import torch
from torch.utils.data import DataLoader, Dataset

from typing import List
## Define a vocabulary class ##

class Vocabulary:
    """
    Vocabulary class.
    """
    def __init__(self, freq_threshold: int):
        self.itos = { 0:'<EOS>', 1:'<UNK>'}
        self.stoi = {'<EOS>': 0, '<UNK>':1}
        self.freq_threshold = freq_threshold
        self.freq_dict = defaultdict(int)
        
    def __len__(self) -> int:
        """
        Get the length of the vocabulary.
        Returns:
            length: the length of the vocabulary.
        """
        length = len(self.itos)
        return length
    
    def buildVocab(self, sentence_list: List[List[str]]) -> None:
        """
        Build the vocabulary from the given sentence list.
        Args:
            sentence_list: the list of sentences.
        """
        idx = len(self.itos)
        for i, sent in enumerate(tqdm(sentence_list)):
            for word in sent:
                self.freq_dict[word] += 1
                
                if self.freq_dict[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numer(self, sent: List[str]) -> List[int]:
        """
        Numericalize the given sentence.
        Args:
            sent: the sentence to be numericalized.
        Returns:
            numer_sent: the numericalized sentence.
        """
        numer_sent = [self.stoi[word] if word in self.stoi else self.stoi['<UNK>'] for word in sent] + [self.stoi['<EOS>']]
        return numer_sent

class custom_dataset(Dataset):
    """
    Custom dataset class.
    """
    def __init__(self, text_data: List[List[str]]):
        """
        Initialize the dataset.
        Args:
            text_data: the list of sentences.
        """
        self.text_data = text_data

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        Returns:
            length: the length of the dataset.
        """
        length = len(self.text_data)
        return length

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get the item at the given index.
        Args:
            index: the index of the item.
        Returns:
            torch_sentence: the numericalized sentence in torch tensor.
        """
        sentence = self.text_data[index]
        torch_sentece = torch.tensor(sentence)
        return torch_sentece

def load_data(path: str, freq_threshold: int = 5, min_len: int = 5, max_len:int = 14):
    """
    Load the data from the given path, and preprocess the data.
    Args:
        path: the path of the data.
        freq_threshold: the frequency threshold of the vocabulary.
        min_len: the minimum length of the sentence.
        max_len: the maximum length of the sentence.
    Returns:

    """
    # load the vocabulary from "eng-fra.txt"
    lines = open(path).read().strip().split('\n')

    # normalize the text
    full_text_data = [[normalizeString(s) for s in l.split('\t')][0] for l in lines]

    # split sentences into words
    full_text_data = [sentence.split() for sentence in full_text_data]
    print(f"Total number of sentences is {len(full_text_data)} \n")

    # create the vocabulary
    vocab = Vocabulary(freq_threshold)
    vocab.buildVocab(full_text_data)
    print(f"The vocabulary size if {len(vocab)} \n")
    
    # filter sentences by length
    new_text = [sent for sent in full_text_data if min_len <= len(sent) <= max_len]
    print(f"The size of the newly created text data is {len(new_text)}, {(len(new_text)*100)//len(full_text_data)}% of the original text")


    data_dict = defaultdict(list)
    for sent in new_text:
        num_sentence = vocab.numer(sent)
        data_dict[len(sent)].append(num_sentence)
    

    return data_dict, vocab


if __name__ == "__main__":
    path = os.path.join("../data", "eng-fra.txt")
    load_data(path)

