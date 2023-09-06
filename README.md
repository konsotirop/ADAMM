# ADAMM: Anomaly Detection of Attributed Multi-graphs with Metadata: A Unified Neural Network Approach

### Prerequisites: [torch](https://pytorch.org/), [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) 

### Graph/Metadata representation: 
Graphs & their associated Metadata features are jointly represented in ADAMM using Pytorch-geometric's [Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object.

The Data object should contain an additional attribute _"metadata"_ for storing the metadata features vector.

ADAMM supports directed, node & edge attributed, multi-graphs.

### Example:
Function _**adamm_fit(train_dataset,test_datest)**_ provides a simple example on how to train & test ADAMM using a custom dataset.</br>

"train_dataset" is a list of torch-geometric Data objects (acting as train dataset), while "test_dataset" is the test dataset used for
scoring anomalies in an inductive setting.

It returns the anomaly scores of the samples in test_dataset and the train validation score used for model selection.
### Configuration:
File _config.json_ allows to modify ADAMM's  hyperparameters and configuration.
* If the input graphs are labeled the field "nfeat_node" should remain None, o.w. should be set equal to number of features.
* If metadata features are used the field "metadata_dim" should be set equal to the dimensions of the metadata, o.w. should be set to None.
