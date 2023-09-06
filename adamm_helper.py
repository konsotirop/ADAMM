import pytorch_lightning as pl
from model import ADAMM
from torch_geometric.loader import DataLoader
from data_processing import ToMultigraph
import json
import torch
import numpy as np

def read_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    if config['nfeat_node'] == "None":
        config['nfeat_node'] = None
    if config["metadata_dim"] == "None":
        config["metadata_dim"] = None
    return config

def adamm_fit(train_dataset, test_dataset):

    description = 'test0'
    config = read_config()
    
    trans = ToMultigraph()
    multiedge_dataset = [trans(d) for d in train_dataset]
    multiedge_test_dataset = [trans(d) for d in test_dataset] 
    train_loader = DataLoader(multiedge_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=6)
    test_loader =  DataLoader(multiedge_test_dataset, batch_size=256, shuffle=False, num_workers=6)

    model = ADAMM(config["nfeat_node"], config["nfeat_edge"],
                                        nhid=config["nhid"], 
                                        nlayer=config["nlayers"],
                                        dropout=config["dropout"], 
                                        learning_rate=config["learning_rate"],
                                        weight_decay=config["weight_decay"],
                                        k_cls=config["k_cls"],
                                        lambda_energy=1,
                                        lambda_diversity= config["diversity"],
                                        lambda_entropy = config["entropy"],
                                        dim_metadata=config["metadata_dim"])

    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=config["epochs"], log_every_n_steps=1, 
                        default_root_dir='results/tb/'+description, check_val_every_n_epoch=15, num_sanity_val_steps=0)

    trainer.fit(model, train_loader, test_loader)

    scores = []
    train_validation_score, test_validation_score = [], []
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            scores.append(model(data)[0])
            test_validation_score.append(model.get_validation_score(data))
        for data in train_loader:
            train_validation_score.append(model.get_validation_score(data))

    scores = torch.cat(scores).detach().numpy()

    return scores, np.mean(train_validation_score), np.mean(test_validation_score)