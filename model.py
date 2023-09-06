import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_scatter import scatter
import pytorch_lightning as pl # for training wrapper
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *

from elements import DiscreteEncoder, MLP, LinearCLIPV2, MLN
import pyg_gnn_wrapper as gnn_wrapper 
import torch.distributions as Dist
from math import pi

class DeepADAMM(pl.LightningModule):
    def __init__(self, model, estimation_layer, learning_rate=0.001, weight_decay=5e-4, **kwargs):
        super().__init__()
        self.model = model
        self.estimation_layer = estimation_layer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay 

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, data):
        # generate anomaly score for each graph
        raise NotImplementedError 


    def validation_step(self, batch, batch_idx): 
        if self.current_epoch > 0:
            self.validation_step_outputs.append((self(batch)[0], batch.y))
            return self(batch)[0], batch.y#.squeeze(-1)

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            # assume label 1 is anomaly and 0 is normal. (pos=anomaly, neg=normal)
            anomaly_scores = torch.cat([out[0] for out in self.validation_step_outputs]).cpu().detach()
            ys = torch.cat([out[1] for out in self.validation_step_outputs]).cpu().detach()
            # import pdb; pdb.set_trace()

            roc_auc = roc_auc_score(ys, anomaly_scores)
            pr_auc = average_precision_score(ys, anomaly_scores)
            avg_score_normal = anomaly_scores[ys==0].mean()
            avg_score_abnormal = anomaly_scores[ys==1].mean()  

            metrics = {'val_roc_auc': roc_auc, 
                       'val_pr_auc': pr_auc, 
                       'val_average_score_normal': avg_score_normal,
                       'val_average_score_anomaly': avg_score_abnormal}
            self.log_dict(metrics, prog_bar=True)
            self.validation_step_outputs.clear() 
        
    def test_step(self, batch, batch_idx): 
        self.test_step_outputs.append((self(batch)[0], batch.y))
        return self(batch), batch.y#.squeeze(-1)

    def on_test_epoch_end(self):
        # assume label 1 is anomaly and 0 is normal. (pos=anomaly, neg=normal)
        anomaly_scores = torch.cat([out[0] for out in self.test_step_outputs]).cpu().detach()
        ys = torch.cat([out[1] for out in self.test_step_outputs]).cpu().detach()
        # import pdb; pdb.set_trace()

        roc_auc = roc_auc_score(ys, anomaly_scores)
        pr_auc = average_precision_score(ys, anomaly_scores)
        avg_score_normal = anomaly_scores[ys==0].mean()
        avg_score_abnormal = anomaly_scores[ys==1].mean()  

        metrics = {'roc_auc': roc_auc, 
                   'pr_auc': pr_auc, 
                   'average_score_normal': avg_score_normal,
                   'average_score_anomaly': avg_score_abnormal}
        self.log_dict(metrics)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.parameters()) + list(self.estimation_layer.parameters()), 
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)


class ADAMM(DeepADAMM):
    def __init__(self, nfeat_node, nfeat_edge,
                 nhid=32, 
                 nlayer=3,
                 dropout=0, 
                 learning_rate=0.001,
                 weight_decay=0,
                 lambda_energy=0.1,
                 lambda_cov_diag=0.0005,
                 lambda_recon=0,
                 lambda_diversity = 0,
                 lambda_entropy = 0,
                 k_cls=2,
                 dim_metadata=None,
                 **kwargs):
        model = MultigraphGNNWrapper(nfeat_node, nfeat_edge, nhid, nhid, nlayer, dim_metadata=dim_metadata, dropout=dropout)
        estimation_layer = MLN(nhid,nhid,k_cls)
        super().__init__(model, estimation_layer, learning_rate, weight_decay)
        self.save_hyperparameters() # self.hparams
        self.radius = 0
        self.nu = 1
        self.eps = 1e-12
        self.register_buffer("phi", torch.zeros(k_cls,))
        self.register_buffer("mu", torch.zeros(k_cls,nhid))
        self.lambda_energy = lambda_energy
        self.lambda_recon = lambda_recon
        self.lambda_diversity = lambda_diversity
        self.lambda_entropy = lambda_entropy
        self.training_step_outputs = []
        self.k_cls = k_cls
        self.metadata_dim = dim_metadata
        
    def get_hiddens(self, data):
        embs = self.model(data)
        return embs
    
    def get_means(self):
        return self.mu
    
    def get_validation_score(self,data):
        s_e,_,_,_ = self.forward(data)
        return torch.mean(s_e).item()

    def forward(self, data):
        if self.metadata_dim is None:
            embs = self.model(data)
        else:
            embs, logits = self.model(data)
            
        # Step1: From embs we get gamma
        gamma = self.estimation_layer(embs)
        # Step 2: We calculate mu,sigma
        if self.training:
            phi, mu = self.compute_gmm_params(embs,gamma)
        else:
            phi = self.phi
            mu = self.mu
        # Step 3: We estimate sample energy
        sample_energy = self.compute_energy(embs, phi, mu, gamma)
        if self.metadata_dim is None:
            return sample_energy, 0, gamma, mu
        else:
            return sample_energy, logits, gamma, mu

    def compute_gmm_params(self, z, gamma):
        
        N = gamma.size(0)
        
        phi = torch.sum(gamma,dim=0) + self.eps
        mu = torch.sum(gamma.unsqueeze(-1)*z.unsqueeze(1), dim=0) / phi.unsqueeze(-1)
        phi = phi / N

        # Register to buffer for forward pass
        with torch.no_grad():
            self.mu = mu
            self.phi = phi
        return phi, mu
    
    def compute_energy(self, z, phi, mu, gamma, size_average=False):
        # Weighted (based on gamma) distance of points from centroids
        #Ci = \sum_{1..K} gamma_ik*||z_i-mu_k||^2
        E = torch.cdist(z,mu).pow(2)*gamma
        E = torch.sum(E,dim=1,keepdim=True)
        return E
        
    
    def training_step(self, batch, batch_idx):
        sample_energy, logits, gamma, mu = self(batch)
        if self.metadata_dim is None:
            clip_loss = 0.
        else:
            with torch.no_grad():
                N = logits.shape[0]
                labels = (2*torch.eye(N) - torch.ones((N,N))).to(logits.device)
            clip_loss = -torch.sum(self.log_sigmoid(labels*logits)) / N
        if self.k_cls > 1 and self.lambda_diversity > 0:
            diversity_loss = -torch.logdet(torch.cov(mu))
            
            entropy_loss = -torch.mean((gamma*torch.log2(gamma)).sum(dim=1))
        elif self.k_cls > 1 and self.lambda_diversity == 0:
            entropy_loss = -torch.mean((gamma*torch.log2(gamma)).sum(dim=1))
            diversity_loss = 0.
        else:
            diversity_loss = 0.
            entropy_loss = 0.
        loss = self.lambda_energy * torch.mean(sample_energy) + self.lambda_recon*clip_loss
        loss = loss + self.lambda_diversity*diversity_loss + self.lambda_entropy*entropy_loss
        with torch.no_grad():
            print("Loss: ", loss.item())
            print("Sample energy: ", self.lambda_energy*torch.mean(sample_energy).item())
            if self.k_cls > 1:
                print("Diversity loss: ", self.lambda_diversity*diversity_loss)
                print("Entropy loss: ",self.lambda_entropy*entropy_loss)
        return loss

    def log_sigmoid(self,x):
        return torch.clamp(x, max=0)-torch.log(torch.exp(-torch.abs(x))+1)



# we change change the conv of BaseGNN to support different version 
class BaseGNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nin, nout, nlayer, dropout=0, gnn_type='GINEConv'):
        super().__init__()
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nin, nin, bias=True) for _ in range(nlayer)]) # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin) for _ in range(nlayer)])
        # self.output_encoder = MLP(nin, nout, nlayer=1, with_final_activation=True, bias=True) 
        self.dropout = dropout

    def reset_parameters(self):
        # self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, x, edge_index, edge_attr, batch):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr, batch)
            x = norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + previous_x 
            previous_x = x
            x = F.relu(x)
        # x = self.output_encoder(x)
        return x

class MultigraphGNNWrapper(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, dim_metadata=None, dropout=0) -> None:
        super().__init__()
        self.node_encoder = DiscreteEncoder(nhid, max_num_values=500) if nfeat_node is None else MLP(nfeat_node, nhid, nlayer=1, with_final_activation=False)
        self.edge_direction_encoder = DiscreteEncoder(nhid, max_num_values=4)
        self.edge_attr_encoder = DiscreteEncoder(nhid, max_num_values=10) if nfeat_edge is None else MLP(nfeat_edge, nhid, nlayer=3, with_final_activation=False)
        self.edge_transform = MLP(nhid, nhid, nlayer=1, with_final_activation=False)
        self.conv_model = BaseGNN(nhid, nhid, nlayer, dropout)
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False, with_norm=True)
        self.graph_metadata_encoder = LinearCLIPV2(nout, dim_metadata, nout, nout, nlayer=2) if dim_metadata else None
        self.reset_parameters()

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_attr_encoder.reset_parameters()
        self.edge_direction_encoder.reset_parameters()
        self.edge_transform.reset_parameters()
        self.conv_model.reset_parameters()
        self.output_encoder.reset_parameters()
        if self.graph_metadata_encoder is not None:
            self.graph_metadata_encoder.reset_parameters()

    def forward(self, data):
        # Encode input nodes and edges
        x = self.node_encoder(data.x)
        edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)) if data.edge_attr is None else data.edge_attr 
        edge_attr = edge_attr if edge_attr.dim() > 1 else edge_attr.unsqueeze(-1)
        edge_attr = self.edge_attr_encoder(edge_attr) + self.edge_direction_encoder(data.edge_direction)
        
        # Transform multi-edge to simple edge with DeepSet
        simplified_edge_attr = scatter(edge_attr, index=data.simplified_edge_batch, dim=0, reduce='add')
        simplified_edge_attr = self.edge_transform(simplified_edge_attr)
        
        # Passing to several conv layers
        x = self.conv_model(x, data.simplified_edge_index, simplified_edge_attr, data.batch)

        # Get embedding of the graph
        x = scatter(x, data.batch, dim=0, reduce='add')
        x = self.output_encoder(x)

        # Get metadata of the graph
        if self.graph_metadata_encoder is not None:
            x, logits = self.graph_metadata_encoder(x, data.metadata)

            return x, logits
        return x
    








if __name__ == '__main__':
    print("code for ADAMM")