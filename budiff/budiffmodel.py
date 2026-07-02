from .nnunet3d_denoise import get_nnunet3d_denoise
from .nnunet3d import get_nnunet3d
import torch.nn as nn 

class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 ddim_steps=3, rand_steps=1, bta=True, ablation_mode="full", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ablation_mode = ablation_mode
        self.edge_model = get_nnunet3d(in_chans=in_channels, out_chans=out_channels)
        self.denoise_model = get_nnunet3d_denoise(in_chans=in_channels, out_chans=out_channels, 
                                          ddim_step=ddim_steps,
                                          rand_step=rand_steps,
                                          bta=bta)

    def set_ablation_mode(self, mode):
        self.ablation_mode = mode

    def forward(self, image, gt=None, ddim=False, ablation_mode=None, return_training_outputs=False):
        mode = ablation_mode or self.ablation_mode
        pred_edge, embeddings = self.edge_model(image)

        if mode == "boundary_only":
            return pred_edge

        use_embeddings = embeddings
        if mode in ["denoise_only", "no_boundary_features", "no_mba", "no_bta"]:
            use_embeddings = None

        if return_training_outputs and gt is not None and not ddim:
            pred, uncertainty = self.denoise_model(
                image,
                gt=gt,
                embeddings=use_embeddings,
                ddim=False
            )
            if mode in ["denoise_only", "no_boundary_logits"]:
                return pred, pred_edge * 0, uncertainty
            return pred, pred_edge, uncertainty

        pred = self.denoise_model(image, gt=gt, 
                                        embeddings=use_embeddings, 
                                        ddim=True)

        if mode in ["denoise_only", "no_boundary_logits"]:
            return pred

        return pred + pred_edge
