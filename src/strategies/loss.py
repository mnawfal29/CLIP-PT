import torch
import torch.nn.functional as F

class CrossDispersionLoss(torch.nn.Module):
    def __init__(self, txt_beta=0):
        super().__init__()
        self.txt_rbf_t = 2.0 # Hyperparameter for text dispersion loss
        self.txt_beta = txt_beta # Weight for text dispersion loss

    def forward(self, logits, targets, text_features):
        """
        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground-truth labels.
            text_features (torch.Tensor): Text feature embeddings.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Cross-Entropy Loss
        loss_ce = F.cross_entropy(logits, targets)

        # Text Prompt Dispersion Loss
        loss_dist_t = torch.pdist(text_features.to(torch.float), p=2).pow(2.0)
        loss_dist_t = loss_dist_t.mul(-self.txt_rbf_t).exp().mean()
        print(loss_ce, loss_dist_t)
        # exit()

        # Combined Loss
        total_loss = loss_ce + self.txt_beta * loss_dist_t
        return total_loss
