from network import Tacotron
from data import get_dataset, DataLoader, collate_fn, get_param_size, inv_spectrogram, find_endpoint, save_wav, spectrogram
from torch import optim
import numpy as np
import argparse
import os
import time
import torch
import io
import torch.nn as nn
from text.symbols import symbols, en_symbols
import hyperparams as hp
from text import text_to_sequence
from torch.utils.tensorboard import SummaryWriter



def generate(model, text, device, writer, curr, _tt):

    # Text to index sequence
    cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
    seq = np.expand_dims(np.asarray(text_to_sequence(text), dtype=np.int32), axis=0)
    
    # Provide [GO] Frame
    mel_input = np.zeros([seq.shape[0], hp.num_mels, 1], dtype=np.float32)
    
    # Variables
    characters = torch.from_numpy(seq).type(torch.cuda.LongTensor).to(device)
    mel_input = torch.from_numpy(mel_input).type(torch.cuda.FloatTensor).to(device)
    mel_input = torch.transpose(mel_input, 1, 2)
    
    # Spectrogram to wav
    _, linear_output = model(characters, mel_input, False)
    linear_output = torch.transpose(linear_output, 1, 2)
    wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
    wav = wav[:find_endpoint(wav)]
    
    wav_tensor = wav * 1.0 / max(0.01, np.max(np.abs(wav)))
    wav_tensor = torch.Tensor(wav).to(device).view(1, -1)

    writer.add_audio('audio_result_%02d'%(_tt), wav_tensor, curr, hp.sample_rate)
    
    out = io.BytesIO()
    save_wav(wav, out)
    
    return out.getvalue()


def main(args):

    # Get dataset
    dataset = get_dataset()
    
    # Construct model
    device = torch.device('cuda:0')
    
    if 'english' in hp.cleaners:
        _symbols = en_symbols
        
    elif 'korean' in hp.cleaners:
        _symbols = symbols
    model = Tacotron(len(_symbols)).to(device)
    
    # Make optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n--------model restored at step %d--------\n" % args.restore_step)

    except:
        print("\n--------Start New Training--------\n")

    # Training
    model = model.train()

    # Make checkpoint directory if not exists
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)
    
    if not os.path.exists(hp.output_path):
        os.mkdir(hp.output_path)
        
    # Tensorboard
    writer = SummaryWriter('runs/tacotron')
    
    sentences = [
    'Scientists at the CERN laboratory say they have discovered a new particle.', 
    'President Trump met with other leaders at the Group of 20 conference.',
    'Generative adversarial network or variational auto-encoder.',
    'Does the quick brown fox jump over the lazy dog?'
    ]

    criterion = nn.L1Loss()

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
    
    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=8)

        for i, data in enumerate(dataloader):

            current_step = i + args.restore_step + epoch * len(dataloader) + 1

            optimizer.zero_grad()

            # Make decoder input by concatenating [GO] Frame
            try:
                mel_input = np.concatenate((np.zeros([args.batch_size, hp.num_mels, 1], dtype=np.float32),data[2][:,:,:-1]), axis=2)
            except:
                raise TypeError("not same dimension")
            
            characters = torch.from_numpy(data[0]).type(torch.cuda.LongTensor).to(device)
            mel_input = torch.from_numpy(mel_input).type(torch.cuda.FloatTensor).to(device)
            mel_spectrogram = torch.from_numpy(data[2]).type(torch.cuda.FloatTensor).to(device)
            linear_spectrogram = torch.from_numpy(data[1]).type(torch.cuda.FloatTensor).to(device)
            
            mel_input = torch.transpose(mel_input, 1, 2)
            mel_spectrogram = torch.transpose(mel_spectrogram, 1, 2)
            linear_spectrogram = torch.transpose(linear_spectrogram, 1, 2)

            # Forward
            mel_output, linear_output = model.forward(characters, mel_input)

            # Calculate loss
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output-linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:,:n_priority_freq])
            loss = mel_loss + linear_loss

            start_time = time.time()

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time
            
            if current_step % hp.save_step == 0:
                model = model.eval()
                
                for _t, text in enumerate(sentences):
                    wav = generate(model, text, device, writer, current_step, _t)
                    path = os.path.join(hp.output_path, 'result_%d_%d.wav' % (current_step, _t+1))
                    with open(path, 'wb') as f:
                        f.write(wav)

                    f.close()
                    print("save wav file at step %d ..." % (current_step))
                
                model = model.train()

            if current_step % hp.log_step == 0:
                print("time per step: %.2f sec" % time_per_step)
                print("At timestep %d" % current_step)
                print("linear loss: %.4f" % linear_loss.item())
                print("mel loss: %.4f" % mel_loss.item())
                print("total loss: %.4f" % loss.item())
                
                writer.add_scalar('train_mel_loss', mel_loss.item(), current_step)
                writer.add_scalar('train_linear_loss', linear_loss.item(), current_step)
                writer.add_scalar('train_loss', loss.item(), current_step)
                
            if current_step % hp.save_step == 0:
                save_checkpoint({'model':model.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    args = parser.parse_args()
    main(args)
