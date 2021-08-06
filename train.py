from ptpl import PyTorchPipeline, seed_everything
from dataset import get_loaders, ds

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


from model import Encoder, Decoder
import random
import numpy as np
from collections import defaultdict

if __name__ == "__main__":

	seed_everything(42)
	# define train_dataloader
	perSentence, vocab = get_loaders()
	vocab_size = len(vocab)

	trainPerSentence = {}
	valPerSentence = {}

	for key, values in perSentence.items():
		arr = values[:]
		random.shuffle(arr)

		val_size = int(0.2 * len(arr))

		val_arr = arr[:val_size]
		train_arr = arr[val_size:]

		trainPerSentence[key] = train_arr[:]
		valPerSentence[key] = val_arr[:]

	del val_arr, train_arr, perSentence
	train_size = sum([len(value) for value in trainPerSentence.values() ])
	val_size = sum([len(value) for value in valPerSentence.values() ])

		
	hidden_dim = 256
	emb_dim = 32
	nfe = 25
	gpu = 2

	hparams = {
		'wb' : True,
		'span_size': 30,
		'hidden_dim': hidden_dim,
		'emb_dim': emb_dim,
		'nfe': nfe,
		'gpu_num': gpu,
		'num_epochs': 100,
		'num_batches' : 256,
		'path2save': "./weights/hidden_size_" + str(hidden_dim) + "_emb_dim_" + str(emb_dim) + "_nfe_" + str(nfe) + ".pt",
		'learning_rate': 1e-3,
	}
	device = torch.device("cuda:" + str(hparams['gpu_num']))

	

	model = nn.Sequential(
		nn.Embedding(vocab_size, hparams['emb_dim']),
		Encoder(hparams['emb_dim'], hparams['hidden_dim']),
		Decoder(hparams['hidden_dim'], vocab_size, hparams['span_size'], device, hparams['nfe']),
	)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

	
	# define PyTorch pipeline
	ptpl = PyTorchPipeline(
		project_name = "gru_node",
		configs = {
			'device': device,
			'criterion': criterion,
			'optimizer': optimizer,
			'train_dataloader': trainPerSentence,
			'val_dataloader': valPerSentence,
			'print_logs': True,
			'wb': hparams['wb'],
		},
		hparams = hparams,
		model = model,
	)
	# train the model
	ptpl.train(num_epochs = hparams['num_epochs'], path2save = hparams['path2save'])


	#### Debugging on one batch ####

	# perSentence, fullVocab = get_loaders()
	# dataset =  ds(perSentence[5])
	# dataloader = DataLoader(dataset,
	# 	batch_size= 32,
	# 	num_workers= 0,
	# 	shuffle = False,
	# )
	# x = next(iter(dataloader))
	# x = x.to(device)
	
	# for i in range(100):
	# 	loss, output_data, _ = ptpl.run(x, "train")

	# loss, output_data, selected_output_data = ptpl.run(x, "test")

	# output_data = output_data.transpose(1, 0).argmax(dim = 2)
	# selected_output_data = selected_output_data.transpose(1, 0).argmax(dim = 2)
	
	# print("Target sentence ", x[0].cpu() )
	# print("Output ", output_data[0].cpu())
	# print("Selected Output ", selected_output_data[0].cpu())