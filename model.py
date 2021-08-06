## PyTorch ##
import torch
import torch.nn as nn
import torch.optim as optim

## TorchDyn ##
from torchdyn.models import *
from torchdyn.datasets import *
from torchdyn import *

from dataset import *

from tqdm import tqdm

class Encoder(nn.Module):
	def __init__(self, emb_dim, hidden_dim):
		super(Encoder, self).__init__()
		self.rnn = nn.GRU(
			emb_dim,
			hidden_dim,
		)
		self.dropout = nn.Dropout(0.3)
	def forward(self, x):
		x = x.transpose(1, 0)
		outputs, hidden = self.rnn(self.dropout(x))
		return self.dropout(hidden)

class Decoder(nn.Module):
	def __init__(self,  hidden_dim, vocab_size, span, device, nfe):
		super(Decoder, self).__init__()
		self.span = torch.linspace(0, span, nfe).to(device)
		func = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.Dropout(0.3),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Dropout(0.3)
		)
		self.ode_solve = NeuralDE(
			func, 
			sensitivity = 'autograd', 
			solver = 'rk4', 
			s_span = self.span,
		)
		self.fc = nn.Linear(hidden_dim, vocab_size)
		self.dropout = nn.Dropout(0.3)
	def forward(self, hidden):
		# hidden: (1, num_batches, hidden_dim)
		hidden = hidden.squeeze()
		trajectory = self.ode_solve.trajectory(hidden, self.span)
		out = self.fc(self.dropout(trajectory))
		return out

if __name__ == "__main__":
	perSentence, fullVocab = get_loaders()
	dataset =  ds(perSentence[5])

	num_batches = 2
	seq_length = 6
	vocab_size = len(fullVocab)
	span = 50
	nfe = 20
	hidden_dim = 64
	emb_dim = 32
	device = torch.device('cuda:0')

	seq2seq = nn.Sequential(
		nn.Embedding(vocab_size, emb_dim),
		Encoder(emb_dim, hidden_dim),
		Decoder(hidden_dim, vocab_size, span, device, nfe),
	)
	seq2seq.to(device)

	x = torch.ones(num_batches, seq_length).long()
	x = x.to(device)
	v = seq2seq(x)
	

	dataloader = DataLoader(dataset,
		batch_size= num_batches,
		num_workers= 0,
		shuffle = False,
	)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(seq2seq.parameters(), lr = 1e-3)

	x = next(iter(dataloader))
	x = x.to(device)

	ints = generate_indices(seq_length, nfe).to(device)

	for i in tqdm(range(100)):
		# x shape is torch.Size([2, 6])
		optimizer.zero_grad()

		target = x.transpose(1, 0)
		# target shape is torch.Size([6, 2])
		y = seq2seq(x)
		# y shape is torch.Size([20, 2, 5384])
		y = y.transpose(1,0)
		# y shape is torch.Size([2, 20, 5384])
		out = y.index_select(1, ints).transpose(1, 0)

		target = target.contiguous().view(seq_length * num_batches, -1).squeeze()
		out = out.contiguous().view(seq_length * num_batches, -1)
		
		loss = criterion(out, target)
		loss.backward()
		#print(loss.item())
		optimizer.step()

	pred = seq2seq(x)
	pred = pred.transpose(1, 0)

	one = pred[0].argmax(dim = 1)
	print(x[0])
	print(one)

	