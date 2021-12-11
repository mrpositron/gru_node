import torch
import torch.nn as nn
from torchdyn.models import NeuralODE

# Define encoder
class Encoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        """
        Define an encoder that takes in a sequence of embeddings and returns a hidden state.
        Args:
            emb_dim: dimension of the embeddings
            hidden_dim: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, emb_dim)
        Returns: 
            hidden: hidden tensor of shape (1, batch_size, hidden_dim)
        """
        x = x.transpose(1, 0) # transform to (seq_len, batch_size, emb_dim)
        _, hidden = self.rnn(self.dropout(x))
        return hidden

# Define decoder 
class Decoder(nn.Module):
    def __init__(self,  hidden_dim: int, vocab_size: int):
        """
        Define NeuralODE decoder that takes in a hidden state and returns trajectory.
        """
        super(Decoder, self).__init__()
        func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3)
        )
        self.ode_solve = NeuralODE(
            func, 
            sensitivity = 'autograd', 
            solver = 'rk4', 
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, hidden: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: Hidden vector of shape (1, batch_size, hidden_dim).
                    It is assumed that the encoder is one directional and has only one layer
            t_span: Interval to evaluate the ODE.
                    Should be a tensor of shape (num_eval_points).
                    It is advised to initialize as follows:
                        t_span = torch.linspace(0, 1, num_eval_points)
        Returns:
            out: Output tensor of shape (num_eval_points, batch_size, vocab_size)
        """
        
        hidden = hidden.squeeze(0)
        trat_eval, trajectory = self.ode_solve(hidden, t_span)
        out = self.fc(self.dropout(trajectory))
        return out



class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        """
        Define a Seq2Seq model with an embedding layer, an encoder and a decoder.
        """
        super(Seq2Seq, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len)
            t_span: Interval to evaluate the ODE.
                    Should be a tensor of shape (num_eval_points).
                    It is advised to initialize as follows:
                        t_span = torch.linspace(0, 1, num_eval_points).
        Returns:
            out: Output tensor of shape (num_eval_points, batch_size, vocab_size)
        """
        x = self.embedding(x)
        hidden = self.encoder(x)
        out = self.decoder(hidden, t_span)
        return out

