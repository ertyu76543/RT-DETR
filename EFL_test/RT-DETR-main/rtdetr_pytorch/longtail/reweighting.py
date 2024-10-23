import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision


class EqualizedFocalLoss(nn.Module):
    def __init__(
        self,
        name="equalized_focal_loss",
        loss_weight=1.0,
        num_classes=1203,
        focal_gamma=2.0,
        focal_alpha=0.25,
        scale_factor=8.0,
    ):
        super().__init__()

        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ignore bg class and ignore idx
        self.num_classes = num_classes

        # cfg for efl loss
        self.scale_factor = scale_factor
        
        # initial variables
        self.register_buffer("pos_grad", torch.zeros(self.num_classes))
        self.register_buffer("neg_grad", torch.zeros(self.num_classes))
        self.register_buffer("pos_neg", torch.ones(self.num_classes))

        self.collect_grad_count = 0 
        self.grad_buffer = []


    def forward(self, input, target, normalizer=None):
        self.input = input.reshape(-1, self.num_classes)
        self.target = target.reshape(-1, self.num_classes)

        pred = torch.sigmoid(self.input)
        pred_t = pred * self.target + (1 - pred) * (1 - self.target) 

        map_val = 1 - self.pos_neg.detach()

        dy_gamma = self.focal_gamma + self.scale_factor * map_val  # r_b + s (1- g^j)

        # focusing factor
        ff = dy_gamma
        # weighting factor
        wf = ff / self.focal_gamma  # r_b + s(1-g^j)/r_b

        # bce_loss
        bce_loss = F.binary_cross_entropy_with_logits(self.input, self.target, reduction="none")
        # alpha_t = self.focal_alpha * self.target + (1 - self.focal_alpha) * (1 - self.target)
        cls_loss = bce_loss * torch.pow((1 - pred_t), ff.detach()) * wf.detach() 

        self.collect_grad(self.target.detach())

        return cls_loss

    def collect_grad(self, target):
        import src.solver.det_engine as det
        grad_in = det.grad_in
        
        if grad_in is not None and self.collect_grad_count % 13 < 5:
            self.collect_grad_count += 1
            
            grad_in = torch.tensor(grad_in)
            grads = grad_in.reshape(-1, self.num_classes)
            grad = torch.abs(grads)
    
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

            if self.pos_grad.device != pos_grad.device:
                self.pos_grad = self.pos_grad.to(pos_grad.device)

            if self.neg_grad.device != neg_grad.device:
                self.neg_grad = self.neg_grad.to(neg_grad.device)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)
            self.grad_buffer = []
            
        elif grad_in is not None and self.collect_grad_count % 13 >= 5:
            self.collect_grad_count += 1
            pass
        else:
            pass