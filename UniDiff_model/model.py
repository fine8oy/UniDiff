import torch
import torch.nn as nn
from contextlib import contextmanager
from UniDiff_model.networks import EDMPrecond
from UniDiff_model.ema import LitEma
from UniDiff_model.img_transformations import STFTEmbedder, DelayEmbedder
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from collections import defaultdict


class UniDiff(nn.Module):
    def __init__(self, args):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps
        self.img_resolution_long = args.img_resolution_long
        self.img_resolution_short = args.img_resolution_short

        self.device = args.device
        self.net = EDMPrecond(args.input_resolution, args.input_channels, channel_mult=args.ch_mult,
                              model_channels=args.unet_channels, attn_resolutions=args.attn_resolution)

        # delay embedding is used
        if not args.use_stft:
            self.delay_long = args.delay_long
            self.embedding_long = args.embedding_long
            self.seq_len_long = args.seq_len_long
            self.ts_img_long = DelayEmbedder(self.device, args.seq_len_long, args.delay_long, args.embedding_long)

            self.delay_short = args.delay_short
            self.embedding_short = args.embedding_short
            self.seq_len_short = args.seq_len_short
            self.ts_img_short = DelayEmbedder(self.device, args.seq_len_short, args.delay_short, args.embedding_short)

        else:
            self.ts_img = STFTEmbedder(self.device, args.seq_len, args.n_fft, args.hop_length)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

        self.tau = args.tau  # very small
        self.lambda_cons = args.lambda_cons

    def ts_to_img(self, signal, pad_val=None):
        """
        Args:
            signal: signal to convert to image
            pad_val: value to pad the image with, if delay embedding is used. Do not use for STFT embedding

        """
        # pad_val is used only for delay embedding, as the value to pad the image with
        # when creating the mask, we need to use 1 as padding value
        # if pad_val is given, it is used to overwrite the default value of 0
        if signal.shape[1] == self.seq_len_long:
            return self.ts_img_long.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img_long.ts_to_img(signal)
        elif signal.shape[1] == self.seq_len_short:
            return self.ts_img_short.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img_short.ts_to_img(signal)

    def img_to_ts(self, img):
        if img.shape[2] == self.img_resolution_long:
            return self.ts_img_long.img_to_ts(img)
        elif img.shape[2] == self.img_resolution_short:
            return self.ts_img_short.img_to_ts(img)

    # init the min and max values for the STFTEmbedder, this function must be called before the training loop starts
    def init_stft_embedder(self, train_loader):
        """
        Args:
            train_loader: training data

        caches min and max values for the real and imaginary parts
        of the STFT transformation, which will be used for normalization.
        """
        assert type(self.ts_img) == STFTEmbedder, "You must use the STFTEmbedder to initialize the min and max values"
        data = []
        for i, data_batch in enumerate(train_loader):
            data.append(data_batch[0])
        self.ts_img.cache_min_max_params(torch.cat(data, dim=0))

    def loss_fn(self, x, kg_cond, img_cond, text_cond):
        '''
        x          : real data if idx==None else perturbation data
        '''

        to_log = {}

        # Forward pass with centered data
        output, weight, D_yn_pert = self.forward(x, kg_cond, img_cond, text_cond)

        # denoising matching term
        loss = (weight * (output - x).square()).mean()
        loss_cons = (output - D_yn_pert).square().mean()
        loss = loss + self.lambda_cons * loss_cons
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def forward(self, x, kg_cond, img_cond, text_cond, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)

        # ---- original code ----
        kg_emb = self.net.model.cond_adapter(kg_cond)
        img_emb = self.net.model.cond_adapter(img_cond)
        text_emb = self.net.model.cond_adapter(text_cond)

        kg_emb_pert = kg_emb + self.tau * torch.randn_like(kg_emb)
        img_emb_pert = img_emb + self.tau * torch.randn_like(img_emb)
        text_emb_pert = text_emb + self.tau * torch.randn_like(text_emb)

        n = torch.randn_like(y) * sigma
        D_yn = self.net(y + n, kg_emb, img_emb, text_emb, sigma, labels, augment_labels=augment_labels)

        D_yn_pert = self.net(y + n, kg_emb_pert, img_emb_pert, text_emb_pert, sigma, labels, augment_labels=augment_labels)

        return D_yn, weight, D_yn_pert

    def pad_f(self, x):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(32, rows)
        padding = (
            0, max_side - rows, 0, 0)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """
        this function updates the EMA model, if it is used
        Args:
            *args:

        Returns:

        """
        if self.use_ema:
            self.model_ema(self.net)
