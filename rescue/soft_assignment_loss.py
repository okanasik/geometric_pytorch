import torch
import torch.nn.functional as F

def soft_assignment_loss(score, data, device):
    score = F.softmax(score, dim=1)
    y_index = torch.argmax(data.y)
    y_pos = data.x[y_index,12:14]
    pos_diff = data.x[:, 12:14]-y_pos
    pos_diff = pos_diff**2
    dist = pos_diff.matmul(torch.tensor([1, 1], dtype=torch.float32).to(device))
    loss1 = (score[:,1]*dist).mean()/data.num_nodes
    loss2 = score[y_index,0]*data.num_nodes
    loss = loss1 + loss2
    return loss