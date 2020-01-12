import torch
import torch.nn as nn
import torch.nn.functional as F



class Pre_net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Pre_net, self).__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.5), 
                                   nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        
    def forward(self, x):
        return self.body(x)
        
class Highway_network(nn.Module):
    def __init__(self, output_hidden = 128, num_layers = 4):
        super(Highway_network, self).__init__()
        
        relu_FC_layer = [nn.Linear(output_hidden, output_hidden)]
        sigmoid_FC_layer = [nn.Linear(output_hidden, output_hidden)]
        
        for i in range(num_layers - 1):
            relu_FC_layer.append(nn.Linear(output_hidden, output_hidden))
            sigmoid_FC_layer.append(nn.Linear(output_hidden, output_hidden))
            
        self.relu_FC_layer = nn.ModuleList(relu_FC_layer)
        self.sigmoid_FC_layer = nn.ModuleList(sigmoid_FC_layer)
        
    def forward(self, x):
        out = x
        
        for relu_layer, sigmoid_layer in zip(self.relu_FC_layer, self.sigmoid_FC_layer):
            relu = F.relu(relu_layer(out))
            sigmoid = torch.sigmoid(sigmoid_layer(out))
            
            out = relu * sigmoid + out * (1 - sigmoid)
            
        return out

    
## For encoder : k = 16, hidden_dim = 128, proj_dim = 128, is_encoder = True
## For decoder : k = 8, hidden_dim = 128, proj_dim = 80, is_encoder = False

class CBHG(nn.Module):
    def __init__(self, k = 16, hidden_dim = 128, proj_dim = 128, is_encoder = True):
        super(CBHG, self).__init__()
        
        self.conv1d_bank = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels = proj_dim, out_channels = hidden_dim, kernel_size = i + 1, stride = 1, padding = (i + 1) // 2), 
                                                      nn.ReLU(inplace=True), nn.BatchNorm1d(num_features = hidden_dim)) for i in range(k)])
        
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride = 1, padding = 2 // 2)
        
        if is_encoder:
            self.conv1d_proj01 = nn.Sequential(nn.Conv1d(in_channels = k * hidden_dim, out_channels = hidden_dim, stride = 1, kernel_size = 3, padding = 3 // 2), 
                                               nn.ReLU(inplace = True), nn.BatchNorm1d(num_features = hidden_dim))
            
            self.conv1d_proj02 = nn.Sequential(nn.Conv1d(in_channels = hidden_dim, out_channels = proj_dim, stride = 1, kernel_size = 3, padding = 3 // 2), 
                                               nn.ReLU(inplace = True), nn.BatchNorm1d(num_features = proj_dim))
            
        else:
            self.conv1d_proj01 = nn.Sequential(nn.Conv1d(in_channels = k * hidden_dim, out_channels = 2 * hidden_dim, stride = 1, kernel_size = 3, padding = 3 // 2), 
                                               nn.ReLU(inplace = True), nn.BatchNorm1d(num_features = 2 * hidden_dim))
            
            self.conv1d_proj02 = nn.Sequential(nn.Conv1d(in_channels = 2 * hidden_dim, out_channels = proj_dim, stride = 1, kernel_size = 3, padding = 3 // 2), 
                                               nn.ReLU(inplace = True), nn.BatchNorm1d(num_features = proj_dim))
        
        self.highway = Highway_network(output_hidden = proj_dim)
        self.GRU = nn.GRU(input_size = proj_dim, hidden_size = hidden_dim, num_layers = 2, batch_first = True, bidirectional = True)
        
    def forward(self, x):
        T = x.size()[-1]
        res = []
        for bank in self.conv1d_bank:
            res.append(bank(x)[:,:,:T])
            
        concat_res = torch.cat(res, dim = 1)
        
        out = self.max_pool(concat_res)[:,:,:T]
        
        out = self.conv1d_proj01(out)
        out = self.conv1d_proj02(out)
        
        highway_input = out + x
        
        highway_input = torch.transpose(highway_input, 1, 2)
        out = self.highway(highway_input)
        
        out, _ = self.GRU(out)
        
        return out

class Attention_Decoder(nn.Module):
    
    def __init__(self, hidden_dim = 256, num_mel = 80, reduction_factor = 5):
        super(Attention_Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_mel = num_mel
        self.reduction_factor = reduction_factor
        
        self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V = nn.Linear(self.hidden_dim, 1)
        self.proj_linear = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.out_linear = nn.Linear(self.hidden_dim, (self.num_mel * self.reduction_factor))
        
        self.Attention_GRU = nn.GRU(self.hidden_dim // 2, self.hidden_dim, batch_first = True)
        self.GRU1 = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first = True)
        self.GRU2 = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first = True)
        
    def forward(self, enc_vec, dec_vec, atten_GRU_h, gru1_h, gru2_h):
        '''
        enc_vec : [bs, T, feature_dim]
        dec_vec : [bs, 1, feature_dim // 2]
        hidden_vec : [1, bs, feature_dim]
        '''
        d_t, next_attention_GRU_h = self.Attention_GRU(dec_vec, atten_GRU_h)
        ## d_t : [bs, 1, feature_dim]
        
        u = self.V(torch.tanh(self.W1(enc_vec) + self.W2(d_t)))
        ## u : [batch_size, T, 1]
        
        u = F.softmax(u, dim = 1)
        u = torch.transpose(u, 1, 2)
        ## u : [batch_size, 1,T]
        
        d_t_dot = torch.bmm(u, enc_vec)
        ## d_t_dot : [batch_size, 1, feature_dim]
        
        concat_feature = torch.cat([d_t, d_t_dot], dim = 2)
        proj_feature = self.proj_linear(concat_feature)
        
        out_gru1, next_gru1_h = self.GRU1(proj_feature, gru1_h)
        
        in_gru2 = out_gru1 + proj_feature
        out_gru2, next_gru2_h = self.GRU2(in_gru2, gru2_h)
        
        out = in_gru2 + out_gru2
        
        out = self.out_linear(out)
        out = out.view(-1, self.reduction_factor, self.num_mel)
        
        return out, next_attention_GRU_h, next_gru1_h, next_gru2_h
        
        
class Encoder(nn.Module):

    def __init__(self, voca_size, emb_dim = 256, hidden_dim = 128, proj_dim = 128):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(voca_size, emb_dim)
        self.pre_net = Pre_net(emb_dim, 2 * hidden_dim)
        self.CBHG = CBHG(k = 16, hidden_dim = hidden_dim, proj_dim = proj_dim, is_encoder = True)
        
    def forward(self, x):
        '''
        Input shape : [bs, T]
        Output shape : [bs, T, feature_dim]
        '''
        
        out = self.embedding(x)
        out = self.pre_net(out)
        out = torch.transpose(out, 1, 2)
        out = self.CBHG(out)
        
        return out
    
    
class Mel_Decoder(nn.Module):
    
    def __init__(self, num_mel = 80, hidden_dim = 256, reduction_factor = 5, max_iter = 200, device = torch.device('cuda:0')):
        super(Mel_Decoder, self).__init__()
        
        self.num_mel = num_mel
        self.hidden_dim = hidden_dim
        self.reduction_factor = reduction_factor
        self.max_iter = max_iter
        self.device = device
        
        self.pre_net = Pre_net(self.num_mel, self.hidden_dim)
        self.atn_decoder = Attention_Decoder(self.hidden_dim, self.num_mel, self.reduction_factor)
        
        
    def forward(self, enc_vec, decoder_input, is_train = True):
        '''
        Input shape : [bs, T, num_mel] (for train) / [bs, 1, num_mel] (for test)
        Output shape : [bs, K, num_mel]
        '''
        
        out = self.pre_net(decoder_input)
        bs = out.size()[0]

        if is_train:
            ##[bs, T, hidden_dim]
            
            iteration_step = out.size()[1] // self.reduction_factor
            
            ## Go frame
            atn_decoder_input = out[:, :1, :]
            atten_GRU_h = torch.zeros(1, bs, self.hidden_dim).to(self.device)
            gru1_h = torch.zeros(1, bs, self.hidden_dim).to(self.device)
            gru2_h = torch.zeros(1, bs, self.hidden_dim).to(self.device)
            outputs = []
            for i in range(iteration_step):
                output, atten_GRU_h, gru1_h, gru2_h = self.atn_decoder(enc_vec, atn_decoder_input, atten_GRU_h, gru1_h, gru2_h)
                
                outputs.append(output)
                
                atn_decoder_input = out[:, (i + 1) * self.reduction_factor : (i + 1) * self.reduction_factor + 1,:]
        
        else:
            
            ## Go frame
            atn_decoder_input = out
            atten_GRU_h = torch.zeros(1, bs, self.hidden_dim).to(self.device)
            gru1_h = torch.zeros(1, bs, self.hidden_dim).to(self.device)
            gru2_h = torch.zeros(1, bs, self.hidden_dim).to(self.device)
            
            outputs = []
            
            for i in range(self.max_iter):    
                output, atten_GRU_h, gru1_h, gru2_h = self.atn_decoder(enc_vec, atn_decoder_input, atten_GRU_h, gru1_h, gru2_h)
                
                outputs.append(output)
                
                atn_decoder_input = self.pre_net(output[:, -1:, :])
            
        outputs = torch.cat(outputs, dim = 1)
        
        return outputs
    
class Post_processing(nn.Module):
    def __init__(self, hidden_dim = 128, proj_dim = 80, num_freq = 1024):
        super(Post_processing, self).__init__()
        self.CBHG = CBHG(k = 8, hidden_dim = hidden_dim, proj_dim = proj_dim, is_encoder = False)
        self.FC = nn.Linear(2 * hidden_dim, num_freq)
        
    def forward(self, x):
        '''
        Input : [bs, num_mel, T]
        Output : [bs, T, num_freq]
        '''
        out = self.CBHG(x)
        out = self.FC(out)
        
        return out

