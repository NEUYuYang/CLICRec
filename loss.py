import torch
class CLICRecInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(CLICRecInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, all):
        pos_sim = torch.sum(anchor * positives, dim=1) / self.temperature
        all_sim = torch.mm(anchor, all.t()) / self.temperature
        loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(all_sim), dim=-1))
        return loss.mean()