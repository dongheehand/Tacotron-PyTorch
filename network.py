import torch
import torch.nn as nn
from module import Encoder, Mel_Decoder, Post_processing, CBHG



class Tacotron(nn.Module):
    def __init__(self, voca_size, emb_dim = 256, enc_hidden_dim = 128, proj_dim = 128, num_mel = 80, dec_hidden_dim = 256, reduction_factor = 5, num_freq = 1024):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(voca_size = voca_size, emb_dim = 256, hidden_dim = enc_hidden_dim, proj_dim = proj_dim)
        self.mel_decoder = Mel_Decoder(num_mel = num_mel, hidden_dim = dec_hidden_dim, reduction_factor = reduction_factor)
        self.post_processing = Post_processing(hidden_dim = enc_hidden_dim, proj_dim = num_mel, num_freq = num_freq)
    
    def forward(self, text, mel, is_train = True):
        enc_vec = self.encoder(text)
        mel_spec = self.mel_decoder(enc_vec, mel, is_train)
        ## mel_spec : [bs, T, num_mel]
        
        mel_transpose = torch.transpose(mel_spec, 1, 2)
        spec = self.post_processing(mel_transpose)
        
        return mel_spec, spec

