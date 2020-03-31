from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid

dataset =  TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = dataset.shuffle()
print(dataset)

print(len(dataset))

print(dataset.num_classes)
print(dataset.num_node_features)

print(dataset[0])
print(dataset[1])

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)
print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)
print(dataset[0].y)
print(max(dataset[0].y))

