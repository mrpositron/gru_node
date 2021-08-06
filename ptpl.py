# This is PyTorch Template for all sorts of thing
from dataset import ds, generate_indices


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm
import wandb

import os
import random

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class PyTorchPipeline:
	def __init__(self, project_name, configs, hparams, model):
		print(f"PyTorch pipeline for {project_name} is set up")
		## assertion checks
		self.configs = configs
		self.model = model
		self.hparams = hparams
		self.device = self.configs['device'] if 'device' in self.configs else torch.device('cpu')

		# set up wandb logging
		self.wb = self.configs["wb"] if 'wb' in self.configs else False
		if self.wb:
			wandb.init(
				project = project_name,
				config = hparams,	
			)
		
		# set training details
		self.criterion = self.configs['criterion'] if 'criterion' in self.configs else None
		self.optimizer = self.configs['optimizer'] if 'optimizer' in self.configs else None

		# set dataloaders
		self.train_dataloader = self.configs['train_dataloader'] if 'train_dataloader' in self.configs else None
		self.val_dataloader = self.configs['val_dataloader'] if 'val_dataloader' in self.configs else None
		self.test_dataloader = self.configs['test_dataloader'] if 'test_dataloader' in self.configs else None


		# et cetera

		self.print_logs = self.configs["print_logs"]


	def predict(self, batch, mode):
		input_data = batch
		input_data = input_data.to(self.device)

		if mode == "train":
			output_data = self.model(input_data)
		else:
			with torch.no_grad():
				output_data = self.model(input_data)

		return output_data


	def compute_loss(self, batch, output_data):
		# assert (self.criterion != None)
		# input_data, target_data = batch
		# target_data = target_data.to(self.device)
		# curr_loss = self.criterion(output_data, target_data)
		# return curr_loss
		nfe = self.hparams['nfe']
		target = batch.to(self.device)
		

		output_data = output_data.transpose(1, 0)
		# output_data.shape = [num_batches, nfe, vocab_size]
		seq_length, num_batches = target.shape[1], target.shape[0]
		ints = generate_indices(seq_length, nfe).to(self.device)
		#print("Ints", ints)
		selected_output_data = temp = output_data.index_select(1, ints).transpose(1, 0)
		# selected_output_data.shape = [seq_length, num_batches, vocab_size]
		target = target.transpose(1, 0)
		# targe.shape = [seq_length, num_batches]
		target = target.contiguous().view(seq_length * num_batches, -1).squeeze()
		selected_output_data = selected_output_data.contiguous().view(seq_length * num_batches, -1)
		loss = self.criterion(selected_output_data, target)
		return loss, temp


	def run(self, batch, mode):
		
		if mode == 'train':
			assert (self.optimizer != None)
			self.optimizer.zero_grad()

		output_data = self.predict(batch, mode)

		curr_loss, selected_output_data = self.compute_loss(batch, output_data)

		if mode == 'train':
			curr_loss.backward()
			nn.utils.clip_grad_norm_(self.model.parameters(),  1.0)
			self.optimizer.step()

		return curr_loss.cpu().item(), output_data, selected_output_data

	def train(self, num_epochs = None, path2save = None):
		if path2save == None:
			print("The path to save the model is not provided. Thus the model weigths will not be saved.")
		assert(self.train_dataloader != None)
		assert(num_epochs != None)
		self.model.train()

		min_loss = 1e10

		for epoch in range(num_epochs):
			self.model.train()
			dset = []
			for key, values in self.train_dataloader.items():
				dataset = ds(values)
				loader = DataLoader(
					dataset = dataset,
					batch_size= self.hparams['num_batches'],
					shuffle= True,
				)
				for _, x in enumerate(loader):
					dset.append(x)
			random.shuffle(dset)
			
			
			cum_loss = .0
			total_cnt = 0
			for i, batch in enumerate(tqdm(dset)):
				batch_size = batch.shape[0]
				total_cnt += batch_size

				curr_loss, output_data, _ = self.run(batch, mode = "train")
				cum_loss += curr_loss

				if self.wb:
					self.wandb_logging(0, "train", [curr_loss])
			if self.wb:
				self.wandb_logging(1, "train", [cum_loss, epoch])
			if self.print_logs:
				print(f"Epoch: {epoch + 1}/{num_epochs} || Loss: {cum_loss/total_cnt}")
			
			val_loss = self.test()
			if val_loss < min_loss:
				min_loss = val_loss
				# save the model
				print(f"Model is saved! with loss {min_loss}")
				if path2save != None:
					self.save(path2save)

	def test(self):
		self.model.eval()
		val_dset = []
		for values in self.val_dataloader.values():
			dataset = ds(values)
			loader = DataLoader(
			dataset = dataset,
				batch_size= self.hparams['num_batches'],
				shuffle= False,
			)
			for _, x in enumerate(loader):
				val_dset.append(x)
		cum_loss = .0
		total_cnt = 0
		for _, batch in enumerate(tqdm(val_dset)):
			batch_size = batch.shape[0]
			total_cnt += batch_size

			curr_loss, output_data, _ = self.run(batch, mode = "val")
			cum_loss += curr_loss
		if self.print_logs:
			print(f"Validation loss is {cum_loss/total_cnt}")
		if self.wb:
			self.wandb_logging(2, "val", [cum_loss])
		return cum_loss

	def save(self, path2save):
		print(f"The model is saved under the path {path2save}")
		torch.save(self.model.state_dict(), path2save)
	
	def load(self, path2load):
		self.model.load_state_dict(torch.load(path2load))
	

	def wandb_logging(self, state, mode, args):
		# state 0 in iteration
		# state 1 in epoch
		if state == 0:
			wandb.log(
					{
					mode + "/iter_loss": args[0],
					}
				)
		elif state == 1:
			wandb.log(
					{
						mode + "/cum_loss": args[0],
						mode + "/num_epochs": args[1],
					}
				)
		elif state == 2:
			wandb.log(
					{
					mode + "/cum_loss": args[0],
					}
				)





if __name__ == "__main__":
	pass