import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

BN = True
# BN = False

class MaskedBN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.bn2d = nn.BatchNorm2d(num_features)
    def reset_parameters(self):
        self.bn.reset_parameters() 
        self.bn2d.reset_parameters() 
    def forward(self, x, mask=None):
        ### apply BN to the last dim
        # 1d: 
        #    x: b x n x d
        # mask: b x n  
        # 2d: 
        #    x: b x n x n x d
        # mask: b x n x n   
        if mask is None:
            if x.dim() == 3:
                return self.bn(x.transpose(1,2)).transpose(1,2)
            if x.dim() == 2:
                return self.bn(x)
            if x.dim() == 4:
                return self.bn2d(x.permute(0,3,1,2)).permute(0,2,3,1)
        # out = x.clone()
        x[mask] = self.bn(x[mask])
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=500): #10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out = out + self.embeddings[i](x[:, i])
        return out

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True, activation_fn="relu", LN='False'):
        super().__init__()
        n_hid = nout
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                     n_hid if i<nlayer-1 else nout, 
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                        or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if not LN else nn.LayerNorm(n_hid) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this
        self.act = getattr(F, activation_fn)

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = self.act(x)  

        # if self.residual:
        #     x = x + previous_x  
        return x 




# ## Linear Version
class LinearCLIPV2(nn.Module):
    def __init__(self, graph_dim, metadata_dim, projection_dim, nout, nlayer):
        super().__init__()
        self.graph_projection = nn.Parameter(torch.randn(graph_dim, projection_dim//2))
        self.metadata_projection = nn.Parameter(torch.randn(metadata_dim, projection_dim//2))
        self.t_prime = nn.Parameter(torch.randn(1,))
        self.b = nn.Parameter(torch.randn(1,))
        self.metadata_encoder = SAE(metadata_dim,projection_dim//2,metadata_dim) 
        self.encoder = MLP(2*projection_dim, nout, nlayer=2, with_final_activation=False, with_norm=True, bias=False)
        self.metadata_dim = metadata_dim

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.metadata_encoder.reset_parameters()
        
    def forward(self, graph_features, metadata_features):
        # graph embedding - linear projection
        graph_features = F.normalize(graph_features @ self.graph_projection, p=2, dim=1)
        # metadata - linear projection
        metadata_features = self.metadata_encoder(metadata_features.view(-1,self.metadata_dim)) 
        metadata_features = F.normalize(metadata_features @ self.metadata_projection, p=2, dim=1)
        # concatenate and use MLP
        x = torch.cat((graph_features,metadata_features),axis=1)
        # Logits
        t = torch.exp(self.t_prime)
        logits = torch.mm(graph_features,torch.transpose(metadata_features, 0, 1))*t + self.b
        return x, logits
    
class LinearCLIP(nn.Module):
    def __init__(self, graph_dim, metadata_dim, projection_dim, nout, nlayer):
        super().__init__()
        self.graph_projection = nn.Parameter(torch.randn(graph_dim, projection_dim))
        self.metadata_projection = nn.Parameter(torch.randn(metadata_dim, projection_dim)) 
        self.encoder = MLP(2*projection_dim, nout, nlayer=2, with_final_activation=False, with_norm=True, bias=False)
        self.metadata_dim = metadata_dim

    def reset_parameters(self):
        self.encoder.reset_parameters()
        
    def forward(self, graph_features, metadata_features):
        # graph embedding - linear projection
        graph_features = F.normalize(graph_features @ self.graph_projection, p=2, dim=1)
        # metadata - linear projection
        metadata_features = F.normalize(metadata_features.view(-1,self.metadata_dim) @ self.metadata_projection, p=2, dim=1)
        # concatenate and use MLP
        x = self.encoder(torch.cat((graph_features,metadata_features),axis=1))
        return x
    
class MLN(nn.Module):
    def __init__(self, nin, nhid, k_gmm):# add parameters):
        super().__init__()
        layers = []
        layers += [nn.Linear(nin,nhid)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.2)]        
        layers += [nn.Linear(nhid,k_gmm)]

        self.estimation = nn.Sequential(*layers)
        self.softmax_layer = nn.Softmax(dim=1)

    def reset_parameters(self):
        for layer in self.estimation:
            layer.reset_parameters()
        self.softmax_layer.reset_parameters()

    def forward(self, x):
        x = self.estimation(x)
        phi = self.softmax_layer(x)
        return phi

class SAE(nn.Module):
    def __init__(self, nin, nhid, nout):# add parameters):
        super().__init__()
        layers = []
        layers += [nn.Linear(nin,nhid)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(nhid,nhid//2)]
        layers += [nn.Tanh()]  
        layers += [nn.Linear(nhid//2,nhid)]
        layers += [nn.Tanh()]  
        layers += [nn.Linear(nhid,nout)]

        self.estimation = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.estimation:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = self.estimation(x)
        return x



