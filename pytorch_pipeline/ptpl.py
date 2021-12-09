# This is PyTorch Template for all sorts of thing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm
import wandb

import os
import sys
import random

sys.path.append('../dataset')
from dataset import *

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


    def predict(self, batch, state):
        """
        Pass the input through the model and return the output.
        Args:
            batch: input batch.
            state: 'train'/'val'/'test'
        Returns:
            output: output of the model.
        """
        input_data = batch
        input_data = input_data.to(self.device)

        num_eval_points = input_data.shape[1]
        t_span = torch.linspace(0, 1, num_eval_points).to(self.device)
        if state == "train":
            output_data = self.model(input_data, t_span)
        else:
            with torch.no_grad():
                output_data = self.model(input_data, t_span)

        return output_data


    def compute_loss(self, batch, output_data):
        """
        Compute loss.
        Args:
            batch: input batch.
            output_data: output of the model.
        Returns:
            loss: loss of the model.
        """
        target = batch.to(self.device)
        target_flattened = target.reshape(-1)
        output_data_flattened = output_data.reshape(-1, output_data.shape[-1])
        loss = self.criterion(output_data_flattened, target_flattened)
        return loss


    def run(self, batch, state):
        """
        Perform a forward pass, compute loss, and backpropagate.
        Args:
            batch: input batch.
            state: 'train'/'val'/'test'
        Returns:
            loss: loss of the model.
        """
        output_data = self.predict(batch, state)

        curr_loss = self.compute_loss(batch, output_data)

        if state == 'train':
            curr_loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(),  1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return curr_loss, output_data

    def train(self, num_epochs = None, path2save = None):
        if path2save == None:
            print("The path to save the model is not provided. Thus the model weigths will not be saved.")
        assert(self.train_dataloader != None)
        assert(num_epochs != None)
        self.model.train()

        min_loss = 1e10

        for epoch in range(num_epochs):
            self.model.train()
            dataloader = []
            for key, values in self.train_dataloader.items():
                dataset = custom_dataset(values)
                temp_dataloader = DataLoader(
                    dataset = dataset,
                    batch_size= self.hparams['num_batches'],
                    shuffle= True,
                )
                for _, x in enumerate(temp_dataloader):
                    dataloader.append(x)
            random.shuffle(dataloader)
            
            
            cum_loss = .0
            total_cnt = 0
            for i, batch in enumerate(tqdm(dataloader)):
                total_cnt += batch.shape[0]

                curr_loss, output_data = self.run(batch, "train")
                cum_loss += curr_loss * batch.shape[0]

                if self.wb:
                    self.wandb_logging(0, "train", [curr_loss])
            if self.wb:
                self.wandb_logging(1, "train", [cum_loss, epoch])
            
            val_loss = self.test()
            if val_loss < min_loss:
                min_loss = val_loss
                # save the model
                if path2save != None:
                    print(f"Model is saved! with loss {min_loss}")
                    self.save(path2save)

            if self.print_logs:
                print(f"Epoch: {epoch + 1}/{num_epochs} \
                        || Training Loss: {cum_loss/total_cnt} \
                        || Validation Loss: {val_loss}")
            

    def test(self):
        self.model.eval()
        # prepare dataloader
        dataloader = []
        for values in self.val_dataloader.values():
            dataset = custom_dataset(values)
            loader = DataLoader(
                dataset = dataset,
                batch_size= self.hparams['num_batches'],
                shuffle= False,
            )
            for _, x in enumerate(loader):
                dataloader.append(x)
        # perform inference
        cum_loss = .0
        total_cnt = 0
        for _, batch in enumerate(tqdm(dataloader)):
            batch_size = batch.shape[0]
            total_cnt += batch_size

            curr_loss, _ = self.run(batch, "val")
            cum_loss += curr_loss
        if self.wb:
            self.wandb_logging(2, "val", [cum_loss])
        return cum_loss/total_cnt

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