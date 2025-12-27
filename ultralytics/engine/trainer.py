# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pandas.io.api import to_pickle
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, optim
from scipy.ndimage import label
import pickle

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)


def generate_masks_from_teacher_tal(t_mask, teacher_preds, mask_type="pyramid"):
    """
    Generate:
      1) Original TAL masks (hard binary per level)
      2) Mask Pyramid (multi-scale OR fusion mask)

    Args:
        t_mask: [B, 8400] boolean mask from TAL.
        teacher_preds: list of three prediction tensors:
            [B, C, 80, 80], [B, C, 40, 40], [B, C, 20, 20]
        mask_type: 
            "original" -> return only original masks
            "pyramid"  -> return only pyramid masks
            "both"     -> return both (default)

    Returns:
        Based on mask_type:
            original_masks: [B,1,80,80], [B,1,40,40], [B,1,20,20]
            pyramid_masks:  [B,1,80,80], [B,1,40,40], [B,1,20,20]
    """

    batch = t_mask.shape[0]

    # Extract spatial sizes dynamically
    spatial_dims = [(p.shape[2], p.shape[3]) for p in teacher_preds]

    # ---------------------------------------------------------
    # 1. ORIGINAL HARD (BINARY) MASKS
    # ---------------------------------------------------------
    original_masks = []
    start = 0

    for h, w in spatial_dims:
        N = h * w
        end = start + N

        # boolean -> float 0/1
        mask = t_mask[:, start:end].float()
        mask = mask.reshape(batch, 1, h, w)

        original_masks.append(mask)
        start = end

    m3, m4, m5 = original_masks  # 80x80, 40x40, 20x20

    # If user wants only original masks
    if mask_type == "original":
        return original_masks

    # ---------------------------------------------------------
    # 2. MASK PYRAMID (MULTI-SCALE OR)
    # ---------------------------------------------------------

    # Step A: Upsample 40→80 and 20→80
    m4_up = F.interpolate(m4, size=(80, 80), mode="nearest")
    m5_up = F.interpolate(m5, size=(80, 80), mode="nearest")

    # Ensure float domain (binary mix safety)
    m3f = m3.float()
    m4f = m4_up.float()
    m5f = m5_up.float()

    # Step B: OR fusion across scales
    # OR rule: if any mask has 1 → output must be 1
    pyramid_80 = torch.maximum(torch.maximum(m3f, m4f), m5f)
    # Equivalent to OR: pyramid_80 = (m3f > 0) | (m4f > 0) | (m5f > 0)

    # Step C: Downscale back to 40 and 20
    pyramid_40 = F.interpolate(pyramid_80, size=(40, 40), mode="nearest")
    pyramid_20 = F.interpolate(pyramid_80, size=(20, 20), mode="nearest")

    pyramid_masks = [pyramid_80, pyramid_40, pyramid_20]

    # If user wants only pyramid
    if mask_type == "pyramid":
        return pyramid_masks

    # Else return both
    return original_masks, pyramid_masks

class AutoNeckFeatureAdaptor(nn.Module):
    def __init__(self, teacher_channels, student_channels):
        super().__init__()

        assert len(teacher_channels) == len(student_channels), \
            "Teacher and student must have same number of neck levels"

        self.adaptors = nn.ModuleList([
            nn.Conv2d(
                in_channels=t_ch,
                out_channels=s_ch,
                kernel_size=1,
                bias=False
            )
            for t_ch, s_ch in zip(teacher_channels, student_channels)
        ])

    def forward(self, teacher_necks):
        return [
            adaptor(t_feat)
            for adaptor, t_feat in zip(self.adaptors, teacher_necks)
        ]

class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss supporting:
    - L2 (MSE)
    - Cosine similarity
    """
    def __init__(
        self,
        loss_type="l2",              # "l2" or "cosine"
        level_weights=None,
        eps=1e-6
    ):
        super().__init__()
        assert loss_type in ["l2", "cosine"]
        self.loss_type = loss_type
        self.level_weights = level_weights
        self.eps = eps

    def _l2_loss(self, s_feat, t_feat, mask=None):
        diff = (s_feat - t_feat) ** 2
        if mask is not None:
            mask = mask.expand_as(diff)
            return (diff * mask).sum() / (mask.sum() + self.eps)
        return diff.mean()

    def _cosine_loss(self, s_feat, t_feat, mask=None):
        # Normalize along channel dimension
        s = F.normalize(s_feat, dim=1, eps=self.eps)
        t = F.normalize(t_feat, dim=1, eps=self.eps)

        # Cosine similarity per spatial location
        cos_sim = (s * t).sum(dim=1, keepdim=True)
        loss = 1.0 - cos_sim

        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + self.eps)
        return loss.mean()

    def forward(self, student_necks, teacher_necks_adapted, masks=None):
        assert len(student_necks) == len(teacher_necks_adapted)
        if masks is not None:
            assert len(masks) == len(student_necks)

        total_loss = 0.0
        num_levels = len(student_necks)

        for i in range(num_levels):
            s_feat = student_necks[i]
            t_feat = teacher_necks_adapted[i]
            mask = masks[i] if masks is not None else None

            if self.loss_type == "l2":
                loss = self._l2_loss(s_feat, t_feat, mask)
            else:
                loss = self._cosine_loss(s_feat, t_feat, mask)

            if self.level_weights is not None:
                loss = loss * self.level_weights[i]

            total_loss += loss

        return total_loss / num_levels

def compute_attention_map(feat, p=2, eps=1e-6):
    """
    feat: [B, C, H, W]
    returns: [B, 1, H, W]
    """
    attn = torch.sum(torch.abs(feat) ** p, dim=1, keepdim=True)
    norm = torch.sqrt(torch.sum(attn ** 2, dim=(2, 3), keepdim=True))
    return attn / (norm + eps)

class AttentionMapDistillationLoss(nn.Module):
    """
    Attention map distillation with:
    - L2 (MSE)
    - Cosine similarity
    """
    def __init__(
        self,
        p=2,
        loss_type="l2",              # "l2" or "cosine"
        level_weights=None,
        eps=1e-6
    ):
        super().__init__()
        assert loss_type in ["l2", "cosine"]
        self.p = p
        self.loss_type = loss_type
        self.level_weights = level_weights
        self.eps = eps

    def _l2_loss(self, A_s, A_t, mask=None):
        diff = (A_s - A_t) ** 2
        if mask is not None:
            return (diff * mask).sum() / (mask.sum() + self.eps)
        return diff.mean()

    def _cosine_loss(self, A_s, A_t, mask=None):
        # Flatten spatial dimensions
        B = A_s.size(0)
        A_s = A_s.view(B, -1)
        A_t = A_t.view(B, -1)

        cos_sim = F.cosine_similarity(A_s, A_t, dim=1)
        loss = 1.0 - cos_sim

        if mask is not None:
            # Mask acts as spatial weighting
            w = mask.view(B, -1).mean(dim=1)
            return (loss * w).mean()

        return loss.mean()

    def forward(self, student_necks, teacher_necks, masks=None):
        assert len(student_necks) == len(teacher_necks)
        if masks is not None:
            assert len(masks) == len(student_necks)

        total_loss = 0.0
        num_levels = len(student_necks)

        for i in range(num_levels):
            A_s = compute_attention_map(student_necks[i], p=self.p, eps=self.eps)
            A_t = compute_attention_map(teacher_necks[i].detach(), p=self.p, eps=self.eps)
            mask = masks[i] if masks is not None else None

            if self.loss_type == "l2":
                loss = self._l2_loss(A_s, A_t, mask)
            else:
                loss = self._cosine_loss(A_s, A_t, mask)

            if self.level_weights is not None:
                loss = loss * self.level_weights[i]

            total_loss += loss

        return total_loss / num_levels

class UnifiedNeckDistillation(nn.Module):
    """
    Unified FPN / neck distillation module implementing:
    - Channel-wise distillation (CWD)
    - Correlation / relational distillation (CRD)
    - Feature distribution matching (MMD)
    - Spatial softmax attention distillation
    - Channel attention (SE-style) distillation
    """
    def __init__(
        self,
        use_cwd=False,
        use_crd=False,
        use_mmd=False,
        use_spatial_att=False,
        use_channel_att=False,
        level_weights=None,
        mmd_kernel="rbf",
        eps=1e-6
    ):
        super().__init__()

        self.use_cwd = use_cwd
        self.use_crd = use_crd
        self.use_mmd = use_mmd
        self.use_spatial_att = use_spatial_att
        self.use_channel_att = use_channel_att

        self.level_weights = level_weights
        self.mmd_kernel = mmd_kernel
        self.eps = eps

    # ---------------------------------------------------------
    # 1. Channel-wise Distillation (CWD)
    # ---------------------------------------------------------
    def _cwd_loss(self, s_feat, t_feat):
        # GAP statistics
        s_mu = s_feat.mean(dim=(2, 3))
        t_mu = t_feat.mean(dim=(2, 3))
        return F.mse_loss(s_mu, t_mu)

    # ---------------------------------------------------------
    # 2. Correlation / Relational Distillation (CRD)
    # ---------------------------------------------------------
    def _crd_loss(self, s_feat, t_feat):
        B, C, H, W = s_feat.shape

        s = s_feat.view(B, C, -1)
        t = t_feat.view(B, C, -1)

        s = F.normalize(s, dim=2)
        t = F.normalize(t, dim=2)

        R_s = torch.bmm(s, s.transpose(1, 2)) / (H * W)
        R_t = torch.bmm(t, t.transpose(1, 2)) / (H * W)

        return F.mse_loss(R_s, R_t)

    # ---------------------------------------------------------
    # 3. Maximum Mean Discrepancy (MMD)
    # ---------------------------------------------------------
    def _rbf_kernel(self, x, y, sigma=1.0):
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist = x_norm - 2 * torch.mm(x, y.t()) + y_norm.t()
        return torch.exp(-dist / (2 * sigma ** 2))

    def _mmd_loss(self, s_feat, t_feat):
        s = s_feat.flatten(2).mean(dim=2)
        t = t_feat.flatten(2).mean(dim=2)

        K_ss = self._rbf_kernel(s, s)
        K_tt = self._rbf_kernel(t, t)
        K_st = self._rbf_kernel(s, t)

        return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()

    # ---------------------------------------------------------
    # 4. Spatial Softmax Attention
    # ---------------------------------------------------------
    def _spatial_attention_loss(self, s_feat, t_feat, mask=None):
        B, _, H, W = s_feat.shape

        A_s = s_feat.abs().sum(dim=1, keepdim=True)
        A_t = t_feat.abs().sum(dim=1, keepdim=True)

        A_s = F.softmax(A_s.view(B, -1), dim=1).view(B, 1, H, W)
        A_t = F.softmax(A_t.view(B, -1), dim=1).view(B, 1, H, W)

        diff = (A_s - A_t) ** 2
        if mask is not None:
            return (diff * mask).sum() / (mask.sum() + self.eps)
        return diff.mean()

    # ---------------------------------------------------------
    # 5. Channel Attention (SE-style)
    # ---------------------------------------------------------
    def _channel_attention_loss(self, s_feat, t_feat):
        s_att = s_feat.mean(dim=(2, 3))
        t_att = t_feat.mean(dim=(2, 3))

        s_att = F.softmax(s_att, dim=1)
        t_att = F.softmax(t_att, dim=1)

        return F.mse_loss(s_att, t_att)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, student_necks, teacher_necks, masks=None):
        assert len(student_necks) == len(teacher_necks)
        if masks is not None:
            assert len(masks) == len(student_necks)

        total_loss = 0.0
        num_levels = len(student_necks)

        for i in range(num_levels):
            s_feat = student_necks[i]
            t_feat = teacher_necks[i].detach()
            mask = masks[i] if masks is not None else None

            level_loss = 0.0

            if self.use_cwd:
                level_loss += self._cwd_loss(s_feat, t_feat)

            if self.use_crd:
                level_loss += self._crd_loss(s_feat, t_feat)

            if self.use_mmd:
                level_loss += self._mmd_loss(s_feat, t_feat)

            if self.use_spatial_att:
                level_loss += self._spatial_attention_loss(s_feat, t_feat, mask)

            if self.use_channel_att:
                level_loss += self._channel_attention_loss(s_feat, t_feat)

            if self.level_weights is not None:
                level_loss = level_loss * self.level_weights[i]

            total_loss += level_loss

        return total_loss / num_levels

class BaseTrainer:
    """
    A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the test set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        # ---- Distillation flags (defaults) ----
        self.args.cls_dist = getattr(self.args, "cls_dist", False)
        self.args.cls_dist_kl = getattr(self.args, "cls_dist_kl", False)
        self.args.dfl_dist = getattr(self.args, "dfl_dist", False)
        self.args.M2D2 = getattr(self.args, "M2D2", False)
        self.args.l2_dist = getattr(self.args, "l2_dist", False)
        self.args.cls_dist = getattr(self.args, "cls_dist", False)
        self.args.cls_fg_mask = getattr(self.args, "cls_fg_mask", False)
        self.args.dfl_fg_mask = getattr(self.args, "dfl_fg_mask", False)
        self.args.l2_fg_mask = getattr(self.args, "l2_fg_mask", False)
        self.args.feat_distill = getattr(self.args, "feat_distill", False)
        self.args.feat = getattr(self.args, "feat", False)
        self.args.feat_att = getattr(self.args, "feat_att", False)
        self.args.feat_oth = getattr(self.args, "feat_oth", False)
        self.args.feat_mask = getattr(self.args, "feat_mask", False) # None or mask
        self.args.loss_ty = getattr(self.args, "loss_ty", "l2")#l2 or "cosine"
        self.args.teacher_model = getattr(self.args,"teacher_model", "")
        self.args.use_cwd= getattr(self.args, "use_cwd", False) # Channel-Wise Distillation (CWD)
        self.args.use_crd= getattr(self.args, "use_crd", False) # Correlation / Relational Distillation (CRD-style)
        self.args.use_mmd= getattr(self.args, "use_mmd", False) # Feature Distribution Matching (MMD)
        self.args.use_spatial_att= getattr(self.args, "use_spatial_att", False) # Spatial Softmax Attention Distillation
        self.args.use_channel_att= getattr(self.args, "use_channel_att", False) # Channel Attention Distillation (SE-style)
        self.args.level_weights= getattr(self.args, "level_weights", [1., 1., 1.]) # channel weights for all feature distillation and cls distillation
        self.args.feature_lambda= getattr(self.args, "feature_lambda", 1.) # feature loss weight multiplier
        self.args.cls_dist_t= getattr(self.args, "cls_dist_t", 1.)
        self.args.cls_alpha= getattr(self.args, "cls_alpha", 1.)
        self.args.m2d2_t= getattr(self.args, "m2d2_t", 1.)
        self.args.m2d2_alpha= getattr(self.args, "m2d2_alpha", 1.)
        self.args.dfl_t= getattr(self.args, "dfl_t", 1.)
        self.args.dfl_alpha= getattr(self.args, "dfl_alpha", 1.)
        self.args.l2_alpha= getattr(self.args, "cls_alpha", 1.)
        self.args.mask_type = getattr(self.args, "mask_type", "original") # original or pyramid
        self.teacher_args = SimpleNamespace(
            mode="train",
            model=self.args.teacher_model,
            data=self.args.data,
            epochs=self.args.epochs,
            time=self.args.time,
            patience=self.args.patience,
            batch=self.args.batch,
            imgsz=self.args.imgsz,
            save=self.args.save,
            save_period=self.args.save_period,
            cache=False,
            device=self.args.device,
            workers=self.args.workers,
            project=self.args.project,
            name=self.args.name,
            exist_ok=self.args.exist_ok,
            pretrained=True,
            optimizer=self.args.optimizer,
            verbose=self.args.verbose,
            seed=self.args.seed,
            deterministic=self.args.deterministic,
            single_cls=self.args.single_cls,
            rect=self.args.rect,
            cos_lr=self.args.cos_lr,
            close_mosaic=self.args.close_mosaic,
            resume=self.args.resume,
            amp=self.args.amp,
            fraction=self.args.fraction,
            profile=self.args.profile,
            freeze=self.args.freeze,
            multi_scale=self.args.multi_scale,
            overlap_mask=self.args.overlap_mask,
            mask_ratio=self.args.mask_ratio,
            dropout=self.args.dropout,
            val=self.args.val,
            split=self.args.split,
            save_json=self.args.save_json,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            half=self.args.half,
            dnn=self.args.dnn,
            plots=self.args.plots,
            source=self.args.source,
            vid_stride=self.args.vid_stride,
            stream_buffer=self.args.stream_buffer,
            visualize=self.args.visualize,
            augment=False,  # Disable augmentation
            agnostic_nms=self.args.agnostic_nms,
            classes=self.args.classes,
            retina_masks=self.args.retina_masks,
            embed=self.args.embed,
            show=self.args.show,
            save_frames=self.args.save_frames,
            save_txt=self.args.save_txt,
            save_conf=self.args.save_conf,
            save_crop=self.args.save_crop,
            show_labels=self.args.show_labels,
            show_conf=self.args.show_conf,
            show_boxes=self.args.show_boxes,
            line_width=self.args.line_width,
            format="torchscript",
            keras=False,
            optimize=False,
            int8=False,
            dynamic=False,
            simplify=True,
            opset=None,
            workspace=None,
            nms=False,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.0,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            nbs=64,
            hsv_h=0.0,  # No hue augmentation
            hsv_s=0.0,  # No saturation augmentation
            hsv_v=0.0,  # No value augmentation
            degrees=0.0,  # No rotation
            translate=0.0,  # No translation
            scale=0.0,  # No scaling
            shear=0.0,  # No shearing
            perspective=0.0,  # No perspective transformation
            flipud=0.0,  # No vertical flipping
            fliplr=0.0,  # No horizontal flipping
            bgr=0.0,  # No color channel swapping
            mosaic=0.0,  # No mosaic augmentation
            mixup=0.0,  # No mixup augmentation
            cutmix=0.0,  # No cutmix augmentation
            copy_paste=0.0,  # No copy-paste augmentation
            copy_paste_mode="flip",
            auto_augment=None,  # Disable auto augmentation
            erasing=0.0,  # No random erasing
            cfg=None,
            tracker="botsort.yaml",
            save_dir="runs/detect/train32",
        )
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        self.teacher = check_model_file_from_stem(self.teacher_args.model)
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        # ---- Distillation CSV Logger ----
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.distill_csv = self.save_dir / f"distill_losses_{timestamp}.csv"
        
        # ---- Distillation loss registry (single source of truth) ----
        self.distill_loss_flags = {
            "cls_distill_loss": self.args.cls_dist,
            "dfl_distill_loss": self.args.dfl_dist,
            "m2d2_distill_loss": self.args.M2D2,
            "box_reg_distill_loss": self.args.l2_dist,
            "feat_l2_loss": self.args.feat,
            "feat_att_loss": self.args.feat_att,
            "feat_other_loss": self.args.feat_oth,
        }

        # ---- Write CSV header dynamically based on enabled losses ----
        active_columns = ["epoch"] + [
            name for name, enabled in self.distill_loss_flags.items() if enabled
        ]

        with open(self.distill_csv, "w") as f:
            f.write(",".join(active_columns) + "\n")

        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)
            # Start console logging immediately at trainer initialization
            self.run_callbacks("on_pretrain_routine_start")

        # self.teacher.args = self.args
        # --------------------------------------------------
        # Feature distillation modules (REGISTERED ONCE)
        # --------------------------------------------------
        if self.args.feat:
            self.neck_adaptor = AutoNeckFeatureAdaptor(teacher_channels= [128, 256, 512] ,student_channels= [64, 128, 256])

            self.fd_loss_fn = FeatureDistillationLoss(
                loss_type=self.args.loss_ty,
                level_weights=self.args.level_weights
            )
        if self.args.feat_att:
            self.att_loss_fn = AttentionMapDistillationLoss(
                loss_type=self.args.loss_ty,                      #l2 or "cosine"
                level_weights= self.args.level_weights
            )
        if self.args.feat_oth:
            self.neck_adaptor = AutoNeckFeatureAdaptor(teacher_channels= [128, 256, 512] ,student_channels= [64, 128, 256])
            self.kd_loss_fn = UnifiedNeckDistillation(
                    use_cwd=self.args.use_cwd, # Channel-Wise Distillation (CWD)
                    use_crd=self.args.use_crd, # Correlation / Relational Distillation (CRD-style)
                    use_mmd=self.args.use_mmd, # Feature Distribution Matching (MMD)
                    use_spatial_att=self.args.use_spatial_att, # Spatial Softmax Attention Distillation
                    use_channel_att=self.args.use_channel_att, # Channel Attention Distillation (SE-style)
                    level_weights= self.args.level_weights
                )
    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.model.feat_distill = self.args.feat_distill

        self.set_model_attributes()

        ckpt_teacher = self.setup_teacher()
        self.teacher.args = self.teacher_args
        self.teacher = self.teacher.to(self.device)
        self.teacher.feat_distill = self.args.feat_distill

        self.set_teacher_attributes()
        for p in self.teacher.parameters():
            p.requires_grad = False
        # self.teacher.eval()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            # ---- Epoch-level distillation loss accumulators (dynamic) ----
            epoch_distill_losses = {
                name: 0.0
                for name, enabled in self.distill_loss_flags.items()
                if enabled
            }

            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")

                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                cls_dist = self.args.cls_dist
                cls_dist_kl = self.args.cls_dist_kl
                dfl_dist = self.args.dfl_dist
                M2D2 = self.args.M2D2
                l2_dist = self.args.l2_dist
                cls_fg_mask = self.args.cls_fg_mask
                dfl_fg_mask = self.args.dfl_fg_mask
                l2_fg_mask = self.args.l2_fg_mask
                feat_distill = self.args.feat_distill
                feat = self.args.feat
                feat_att = self.args.feat_att
                feat_oth = self.args.feat_oth
                feat_mask = self.args.feat_mask # None or mask
                
                # Forward
            
                with autocast(self.amp):
        
                    batch = self.preprocess_batch(batch)
                    # Save
                    # with open("batch.pkl", "wb") as f:
                    #     pickle.dump(batch, f)
                    progress = (epoch + 1) / self.args.epochs
                    # Forward pass through student and teacher
                    student_out = self.model(batch)
                    
                    with torch.no_grad():
                        teacher_out = self.teacher(batch)
                    if feat_distill:
                        teacher_preds, teacher_necks = teacher_out[1]
                        student_preds, student_necks = student_out[1]
                    else:
                        teacher_preds = teacher_out[1]
                        student_preds = student_out[1]
                
                    _, _, target_labels, target_bboxes, target_scores, t_mask, target_gt_idx, norm_align_metric = teacher_out[0]
                    # TAL based mask for distillation
                    mask = generate_masks_from_teacher_tal(t_mask, teacher_preds, mask_type= self.args.mask_type)
                
                    # Standard YOLO loss & predictions
                    loss, self.loss_items, _ ,_,_,_,_,_= student_out[0]
                    self.loss = loss.sum()
                    # raw feature distilation starts here
                    if feat:

                        teacher_necks_adapted = self.neck_adaptor(
                            teacher_necks,
                        )

                        fd_loss = self.fd_loss_fn(
                            student_necks,
                            teacher_necks_adapted,
                            masks= mask if feat_mask else None
                        )

                        self.loss += self.args.feature_lambda * fd_loss
                        epoch_distill_losses["feat_l2_loss"] += fd_loss.item()

                    # --------------------------------------------------
                    # Attention map distillation
                    # --------------------------------------------------
                    if feat_att:
                        
                        att_loss = self.att_loss_fn(
                            student_necks,
                            teacher_necks,
                            masks=mask if feat_mask else None
                        )

                        self.loss += self.args.feature_lambda * att_loss
                        epoch_distill_losses["feat_att_loss"] += att_loss.item()

                    if feat_oth :
                
                        teacher_necks_adapted = self.neck_adaptor(
                            teacher_necks,
                        )
                        
                        kd_loss = self.kd_loss_fn(
                            student_necks,
                            teacher_necks_adapted,
                            masks=mask if feat_mask else None
                        )
                        epoch_distill_losses["feat_other_loss"] += kd_loss.item()
                        self.loss += self.args.feature_lambda * kd_loss

                    if cls_dist:
                        # --- class Distillation setup ---
                        class_channels = self.data['nc']                                    
         
                        distill_cls_loss = 0.0

                        for i, (s_pred, t_pred) in enumerate(zip(student_preds, teacher_preds)):
                            # --- Extract class logits ---
                            s_logits = s_pred[:, -class_channels:, :, :]  # [B, C, H, W]
                            t_logits = t_pred[:, -class_channels:, :, :]  # [B, C, H, W]

                            if not cls_dist_kl:
                                ############################################################
                                #  ORIGINAL BCE-SIGMOID DISTILLATION
                                ############################################################
                                with torch.no_grad():
                                    t_probs = torch.sigmoid(t_logits / self.args.cls_dist_t)

                                bce = F.binary_cross_entropy_with_logits(
                                    s_logits, t_probs, reduction='none'
                                )  # [B, C, H, W]

                                if cls_fg_mask:
                                    fg_mask = mask[i]  # [B,1,H,W]
                                    masked_bce = bce * fg_mask
                                    active_elems = fg_mask.sum() * class_channels + 1e-6
                                    loss_ = masked_bce.sum() / active_elems
                                else:
                                    loss_ = bce.mean()

                            else:
                                ############################################################
                                #  NEW KL-SOFTMAX DISTILLATION OVER CLASS DIMENSION
                                ############################################################
                                # Softmax across class channels
                                # shape: [B, C, H, W]
                                with torch.no_grad():
                                    t_probs = F.softmax(t_logits / self.args.cls_dist_t, dim=1)

                                s_log_probs = F.log_softmax(s_logits / self.args.cls_dist_t, dim=1)

                                # KL across class dimension (per pixel)
                                # kl: [B, H, W]
                                kl = F.kl_div(s_log_probs, t_probs, reduction='none')  # shape [B, C, H, W]
                                kl = kl.sum(dim=1)  # sum over class dimension → [B, H, W]

                                # Temperature correction (standard in Hinton KD)
                                kl = kl * (self.args.cls_dist_t * self.args.cls_dist_t)

                                if cls_fg_mask:
                                    fg_mask = mask[i]  # [B,1,H,W]
                                    masked_kl = kl * fg_mask
                                    active_elems = fg_mask.sum() + 1e-6
                                    loss_ = masked_kl.sum() / active_elems
                                else:
                                    loss_ = kl.mean()

                            # Weighted accumulation
                            distill_cls_loss += self.args.level_weights[i] * loss_

                        # --- Final loss combination ---
                        self.loss += self.args.cls_alpha * distill_cls_loss
                        
                        # Optional logging
                        epoch_distill_losses["cls_distill_loss"] += (
                            self.args.cls_alpha * distill_cls_loss.item()
                        )

                    if M2D2 == True:
                        # flatten each tensor to [4, -1]
                        flat_masks = [m.view(m.size(0), -1) for m in mask]

                        # concatenate along dimension 1 -> [4, 8400]
                        unified = torch.cat(flat_masks, dim=1)
                        # print(unified.shape)
                        shapes=((80,80),(40,40),(20,20))
                        B, N = t_mask.shape
                        mask_splits = torch.split(unified.bool().cpu(), (6400, 1600, 400), dim=1)
                        label_splits = torch.split(target_labels.cpu(), (6400, 1600, 400), dim=1)
                        updated_preds = []
                        
                        for lvl, feat in enumerate(teacher_preds):
                            # Save device and dtype to reconstruct final tensors
                            device = feat.device
                            dtype = feat.dtype
                            b, C, H, W = feat.shape
                            # Work with a clone to avoid modifying input in-place
                            feat_flat = feat.view(b, C, -1).clone()  # [B, C, HW]
                            cls_part = feat_flat[:, -self.data['nc']:, :]     # [B, num_classes, HW]
                            dfl_part = feat_flat[:, :C - self.data['nc'], :]     # [B, dfl_ch, HW]
                            
                            # For CPU labeling we need flattened length and shapes
                            HW = H * W
                            # We'll do replacements in-place on dfl_part (which is on `device`)
                            # Loop per batch image
                            mask_flat_cpu = mask_splits[lvl].cpu()    # [B, HW] on CPU
                            label_flat_cpu = label_splits[lvl].cpu()  # [B, HW] on CPU
                            for bi in range(B):
                                # get 1/0 mask as numpy 2D for scipy.label
                                mask_1d = mask_flat_cpu[bi].numpy().astype(np.uint8)   # [HW]
                                if mask_1d.sum() == 0:
                                    continue
                                Hs, Ws = shapes[lvl]
                                mask_2d = mask_1d.reshape(Hs, Ws)
                                labeled_array, num_features = label(mask_2d)
                                
                                if num_features == 0:
                                    continue
                                
                                # flattened arrays for class lookup
                                labels_np = label_flat_cpu[bi].numpy()  # [HW], int class ids
                                labeled_flat = labeled_array.reshape(-1)  # [HW]
                                
                                # Iterate components
                                for comp_id in range(1, num_features + 1):
                                    comp_idx_np = np.nonzero(labeled_flat == comp_id)[0]  # indices into flattened HW
                                    if comp_idx_np.size == 0:
                                        continue
                                    
                                    # classes present in this component
                                    classes_in_comp = np.unique(labels_np[comp_idx_np])
                                    
                                    # For each class, find indices inside component with that class
                                    for cls in classes_in_comp:
                                        # boolean selection inside comp for this class
                                        sel_mask = (labels_np[comp_idx_np] == cls)
                                        if not np.any(sel_mask):
                                            continue
                                        cls_positions_np = comp_idx_np[sel_mask]  # numpy indices (1D)
                                        # convert to torch LongTensor on device for indexing
                                        pos_idx = torch.from_numpy(cls_positions_np).long().to(device)  # [K]

                                        if pos_idx.numel() == 0:
                                            continue
                                        
                                        # Gather DFL vectors at these positions: shape [dfl_ch, K]
                                        # dfl_part[bi]: [dfl_ch, HW]
                                        vals = dfl_part[bi][:, pos_idx]  # [dfl_ch, K]

                                        # compute average vector over K -> [dfl_ch, 1]
                                        avg_vec = vals.mean(dim=1, keepdim=True)  # [dfl_ch, 1]

                                        # Broadcast-assign averaged vector to all those positions
                                        dfl_part[bi][:, pos_idx] = avg_vec  # replaced in-place
                            # Reconstruct the feature map: concat cls + updated dfl
                            new_feat_flat = torch.cat([cls_part, dfl_part], dim=1)  # [B, C, HW]
                            new_feat = new_feat_flat.view(b, C, H, W).to(device=device, dtype=dtype)
                            updated_preds.append(new_feat)
                        # ----- M2D2 distillation starts here -----
                        reg_max = 16

                        dfl_distill_loss = 0.0
                        count = 0

                        for scale_idx in range(len(student_preds)):
                            sp = student_preds[scale_idx]      # [B, Ctot, H, W]
                            tp = updated_preds[scale_idx]

                            B, _, H, W = sp.shape
                            dfl_channels = 4 * reg_max

                            # Extract DFL logits (first 64 channels)
                            sp_dfl = sp[:, :dfl_channels, :, :]   # [B, 64, H, W]
                            tp_dfl = tp[:, :dfl_channels, :, :]   # [B, 64, H, W]

                            # Reshape to [B, 4, reg_max, H, W]
                            sp_reshaped = sp_dfl.view(B, 4, reg_max, H, W)
                            tp_reshaped = tp_dfl.view(B, 4, reg_max, H, W)

                            # Compute KL divergence per location and coordinate
                            with torch.no_grad():
                                tp_soft = torch.softmax(tp_reshaped / self.args.m2d2_t, dim=2)  # [B, 4, reg_max, H, W]
                            sp_logsoft = torch.log_softmax(sp_reshaped / self.args.m2d2_t, dim=2)  # [B, 4, reg_max, H, W]

                            # KL divergence: [B, 4, H, W] (sum over reg_max)
                            kl_per_pixel = torch.sum(tp_soft * (torch.log(tp_soft + 1e-8) - sp_logsoft), dim=2)  # [B, 4, H, W]

                            # Reduce over the 4 coordinates (mean or sum)
                            kl_spatial = kl_per_pixel.mean(dim=1)  # [B, H, W]  (you can also use .sum(dim=1))

                            # --- Apply foreground mask if enabled ---
                            if dfl_fg_mask:
                                fg_mask = mask[scale_idx]  # [B, H, W], from your precomputed list
                                masked_kl = kl_spatial * fg_mask.squeeze(1)
                                # Normalize by number of active elements (not just batchmean)
                                active = fg_mask.squeeze(1).sum() + 1e-6
                                loss_kl = masked_kl.sum() / active
                            else:
                                # Original: mean over all pixels and batch
                                loss_kl = kl_spatial.mean()

                            # Scale by T^2 (standard in distillation)
                            loss_kl = loss_kl * (self.args.m2d2_t ** 2)

                            dfl_distill_loss += loss_kl
                            count += 1

                        # Average across scales
                        if count > 0:
                            dfl_distill_loss = dfl_distill_loss / count

                        # Final weight
                        dfl_distill_loss = self.args.m2d2_alpha * dfl_distill_loss

                        # Add to total loss
                        self.loss += dfl_distill_loss

                        # Accumulate epoch-level DFL KD loss
                        epoch_distill_losses["m2d2_distill_loss"] += dfl_distill_loss.item()
                        # M2D2 distillation ends here                
                    if dfl_dist == True:
                        # ----- DFL distillation starts here -----
                        reg_max = 16
                        T = 1.0 

                        dfl_distill_loss = 0.0
                        count = 0

                        for scale_idx in range(len(student_preds)):
                            sp = student_preds[scale_idx]      # [B, Ctot, H, W]
                            tp = teacher_preds[scale_idx]

                            B, _, H, W = sp.shape
                            dfl_channels = 4 * reg_max

                            # Extract DFL logits (first 64 channels)
                            sp_dfl = sp[:, :dfl_channels, :, :]   # [B, 64, H, W]
                            tp_dfl = tp[:, :dfl_channels, :, :]   # [B, 64, H, W]

                            # Reshape to [B, 4, reg_max, H, W]
                            sp_reshaped = sp_dfl.view(B, 4, reg_max, H, W)
                            tp_reshaped = tp_dfl.view(B, 4, reg_max, H, W)

                            # Compute KL divergence per location and coordinate
                            with torch.no_grad():
                                tp_soft = torch.softmax(tp_reshaped / self.args.dfl_t, dim=2)  # [B, 4, reg_max, H, W]
                            sp_logsoft = torch.log_softmax(sp_reshaped / self.args.dfl_t, dim=2)  # [B, 4, reg_max, H, W]

                            # KL divergence: [B, 4, H, W] (sum over reg_max)
                            kl_per_pixel = torch.sum(tp_soft * (torch.log(tp_soft + 1e-8) - sp_logsoft), dim=2)  # [B, 4, H, W]

                            # Reduce over the 4 coordinates (mean or sum)
                            kl_spatial = kl_per_pixel.mean(dim=1)  # [B, H, W]  (you can also use .sum(dim=1))

                            # --- Apply foreground mask if enabled ---
                            if dfl_fg_mask:
                                fg_mask = mask[scale_idx]  # [B, H, W], from your precomputed list
                                masked_kl = kl_spatial * fg_mask.squeeze(1)
                                # Normalize by number of active elements (not just batchmean)
                                active = fg_mask.squeeze(1).sum() + 1e-6
                                loss_kl = masked_kl.sum() / active
                            else:
                                # Original: mean over all pixels and batch
                                loss_kl = kl_spatial.mean()

                            # Scale by T^2 (standard in distillation)
                            loss_kl = loss_kl * (self.args.dfl_t ** 2)

                            dfl_distill_loss += loss_kl
                            count += 1

                        # Average across scales
                        if count > 0:
                            dfl_distill_loss = dfl_distill_loss / count

                        # Final weight
                        dfl_distill_loss = self.args.dfl_alpha * dfl_distill_loss

                        # Add to total loss
                        self.loss += dfl_distill_loss

                        # Accumulate epoch-level DFL KD loss
                        epoch_distill_losses["dfl_distill_loss"] += dfl_distill_loss.item()
                        # dfl distillation ends here
                    if l2_dist == True:
                        # ----- L2 box regression distillation starts here -----
                        reg_max = 16
                        
                        box_reg_loss = 0.0
                        count = 0

                        # Precompute bins once (shape: [1, reg_max])
                        bins = torch.arange(reg_max, device=self.device, dtype=torch.float32).view(1, 1, reg_max, 1, 1)  # [1,1,16,1,1]  # [1, reg_max]
                    
                        for scale_idx in range(len(student_preds)):
                            sp = student_preds[scale_idx]      # [B, Ctot, H, W]
                            tp = teacher_preds[scale_idx]

                            B, Ctot, H, W = sp.shape
                            dfl_channels = 4 * reg_max

                            # Extract DFL logits (first 64 channels)
                            sp_dfl = sp[:, :dfl_channels, :, :]   # [B, 64, H, W]
                            tp_dfl = tp[:, :dfl_channels, :, :]   # [B, 64, H, W]

                            # Reshape to [B, 4, reg_max, H, W]
                            sp_reshaped = sp_dfl.view(B, 4, reg_max, H, W)
                            tp_reshaped = tp_dfl.view(B, 4, reg_max, H, W)

                            # Convert to probabilities
                            sp_prob = torch.softmax(sp_reshaped, dim=2)  # [B, 4, reg_max, H, W]
                            tp_prob = torch.softmax(tp_reshaped, dim=2)  # [B, 4, reg_max, H, W]

                            # Compute expected values (continuous offsets)
                            
                            sp_val = (torch.sum(sp_prob * bins, dim=2) / reg_max)
                            tp_val = (torch.sum(tp_prob * bins, dim=2) / reg_max)

                            # Compute squared error per coordinate → [B, 4, H, W]
                            sq_error = (sp_val - tp_val) ** 2

                            # Reduce over the 4 box sides (mean or sum); we use mean
                            l2_spatial = sq_error.mean(dim=1)  # [B, H, W]

                            # --- Apply foreground mask if enabled ---
                            if l2_fg_mask:
                                fg_mask = mask[scale_idx].squeeze(1)  # [B, H, W]
                                masked_l2 = l2_spatial * fg_mask
                                active = fg_mask.sum() + 1e-6
                                loss_reg = masked_l2.sum() / active
                            else:
                                loss_reg = l2_spatial.mean()

                            box_reg_loss += loss_reg
                            count += 1

                        # Average across scales
                        if count > 0:
                            box_reg_loss = box_reg_loss / count

                        # Apply final weight
                        box_reg_loss = self.args.l2_alpha * box_reg_loss

                        self.loss += box_reg_loss

                        # Accumulate epoch-level box regression KD loss
                        epoch_distill_losses["box_reg_distill_loss"] += box_reg_loss.item()
                        #regression distillation ends here
                    # self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
            # Log epoch-level distillation losses
            if RANK in {-1, 0}:
                if epoch_distill_losses:
                    log_str = ", ".join(
                        f"{k}={v:.6f}" for k, v in epoch_distill_losses.items()
                    )
                    LOGGER.info(f"Epoch {epoch + 1}: {log_str}")

            if RANK in {-1, 0}:  # Only main process logs to file
                row = [str(epoch + 1)] + [
                    f"{epoch_distill_losses[name]:.6f}"
                    for name, enabled in self.distill_loss_flags.items()
                    if enabled
                ]

                with open(self.distill_csv, "a") as f:
                    f.write(",".join(row) + "\n")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] == "ndjson":
                # Convert NDJSON to YOLO format
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo

                yaml_path = asyncio.run(convert_ndjson_to_yolo(self.args.data))
                self.args.data = str(yaml_path)
                data = check_det_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def setup_teacher(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.teacher, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.teacher, None
        ckpt = None
        if str(self.teacher).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.teacher)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.teacher = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Run validation on test set using self.validator.

        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Return a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Return dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def set_teacher_attributes(self):
        """Set or update model parameters before training."""
        self.teacher.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        # --------------------------------------------
        # Add feature distillation adaptor parameters
        # --------------------------------------------
        if hasattr(self, "neck_adaptor"):
            optimizer.add_param_group(
                {
                    "params": self.neck_adaptor.parameters(),
                    "weight_decay": decay,
                }
            )

        # >>> ADD THIS BLOCK HERE <<<
        if hasattr(self, "neck_adaptor"):
            found = False
            target = self.neck_adaptor.adaptors[0].weight
            for pg in optimizer.param_groups:
                if any(p is target for p in pg["params"]):
                    print("Adaptor LR:", pg["lr"])
                    found = True
            assert found, "Neck adaptor NOT in optimizer param groups"
        # >>> END BLOCK <<<
        
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
