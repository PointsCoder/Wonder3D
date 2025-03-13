import torch

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]
    
class NormalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, mask_loss_weights = 0.2):
        super().__init__()
        self.s_l1 = torch.nn.L1Loss()
        self.mask_loss_weights  = mask_loss_weights
        if self.mask_loss_weights > 0.:
            self.mask_loss = torch.nn.MSELoss()
            
    def forward(self, render_normal, gt_normal, weights = None, mask = None, gtmask = None):
        if weights == None:
            weights = torch.ones_like(render_normal[...,[2]])

        t_mask = (mask > 0.5)[...,0].detach()
        # loss = (render_normal[t_mask] - gt_normal[t_mask]).abs().pow(2).mean()
        loss = ((render_normal[...,:3] - gt_normal[...,:3])*weights)[t_mask].abs().pow(2).mean()
        if mask is not None and gtmask is not None and self.mask_loss_weights > 0.:
            n_gtmask = (gtmask < 0.5)[...,0]
            # loss_alpha_target_mask_l2 = (render_normal[..., -1][n_gtmask] - gt_normal[..., -1][n_gtmask]).abs().pow(2).mean()
            loss_alpha_target_mask_l2 = (render_normal[..., -1][n_gtmask] - gt_normal[..., -1][n_gtmask]).abs().mean()
            loss += loss_alpha_target_mask_l2 * self.mask_loss_weights
            
        return loss
