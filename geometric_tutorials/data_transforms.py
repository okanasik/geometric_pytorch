from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
print(dataset[0])

# convert point cloud dataset into a graph dataset, create a nearest neighbor graph via transforms
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6))
print(dataset[0])

# add random noise to each node
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6), transform=T.RandomTranslate(0.01))
print(dataset[0])