import sys
import os

from dataset.dataset import *
from pytorch_pipeline.ptpl import PyTorchPipeline
from model.model import *

import torch
import torch.nn as nn

import random
from tqdm import tqdm


if __name__ == "__main__":
    hidden_dim = 64
    emb_dim = 32


    hparams = {
        'hidden_dim': hidden_dim,
        'emb_dim': emb_dim,
        'num_epochs': 3,
        'num_batches' : 32,
        'path2save': "./weights/hidden_size_" + str(hidden_dim) + "_emb_dim_" + str(emb_dim) + ".pt",
        'learning_rate': 1e-3,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare data

    path = os.path.join("./data", "eng-fra.txt")
    perSentence, vocab = load_data(path)
    vocab_size = len(vocab)

    train_data = {}
    val_data = {}

    for key, values in perSentence.items():
        arr = values[:]
        random.shuffle(arr)

        val_size = int(0.2 * len(arr))

        val_arr = arr[:val_size]
        train_arr = arr[val_size:]

        train_data[key] = train_arr[:]
        val_data[key] = val_arr[:]

    del val_arr, train_arr, perSentence
    train_size = sum([len(value) for value in train_data.values() ])
    val_size = sum([len(value) for value in val_data.values() ])


    print("Size of the training data: ", train_size)
    print("Size of the validation data: ", val_size)
    print()

    # define a model

    embedding = nn.Embedding(vocab_size, emb_dim)
    encoder = Encoder(emb_dim, hidden_dim)
    decoder = Decoder(hidden_dim, vocab_size)
    model = Seq2Seq(embedding, encoder, decoder)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    ptpl = PyTorchPipeline(
        project_name = "gru_node",
        configs = {
            'device': device,
            'criterion': criterion,
            'optimizer': optimizer,
            'train_dataloader': train_data,
            'val_dataloader': val_data,
            'print_logs': True,
            'wb': True,
        },
        hparams = hparams,
        model = model,
    )

    ptpl.train(num_epochs=hparams['num_epochs'], path2save= hparams['path2save'])

