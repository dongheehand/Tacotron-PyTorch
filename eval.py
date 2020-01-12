from network import Tacotron
from data import get_dataset, DataLoader, collate_fn, get_param_size, inv_spectrogram, find_endpoint, save_wav, spectrogram
import numpy as np
import argparse
import os
import torch
import io
from text.symbols import symbols, en_symbols
import hyperparams as hp
from text import text_to_sequence

def main(args):

    device = torch.device('cuda:0')
    
    if 'english' in hp.cleaners:
        _symbols = en_symbols
        
    elif 'korean' in hp.cleaners:
        _symbols = symbols

    model = Tacotron(len(_symbols)).to(device)


    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    model = model.eval()
    
    sentences = [
    'Scientists at the CERN laboratory say they have discovered a new particle.', 
    'President Trump met with other leaders at the Group of 20 conference.',
    'Generative adversarial network or variational auto-encoder.',
    'Does the quick brown fox jump over the lazy dog?'
    ]

    # Text to index sequence

    for i, ele in enumerate(sentences):
        cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
        seq = np.expand_dims(np.asarray(text_to_sequence(ele), dtype=np.int32), axis=0)

        # Provide [GO] Frame
        mel_input = np.zeros([seq.shape[0], hp.num_mels, 1], dtype=np.float32)

        # Variables
        characters = torch.from_numpy(seq).type(torch.cuda.LongTensor).to(device)
        mel_input = torch.from_numpy(mel_input).type(torch.cuda.FloatTensor).to(device)
        mel_input = torch.transpose(mel_input, 1, 2)

        # Spectrogram to wav
        mel_output, linear_output = model(characters, mel_input, False)

        linear_output = torch.transpose(linear_output, 1, 2)
        wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
        _wav = wav[:find_endpoint(wav)]
        out = io.BytesIO()
        save_wav(_wav, out)
        
        tt = out.getvalue()

        f = open('./result_%02d.wav'%i, 'wb')
        f.write(tt)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to restore checkpoint')
    args = parser.parse_args()
    main(args)

