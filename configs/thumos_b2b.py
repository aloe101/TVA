_base_ = [
    "_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",  # dataset config
]

window_size = 768   #768
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        test_mode=False,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
)

model_sur = dict(backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="/tmp/tad/OpenTAD/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth",   #"/tmp/tad/AttackTAD/pretrained/new_epoch_6.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                # dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b t (t1 c)", t1=chunk_num),
                # dict(type="Reduce", keys=["feats"], ops="b t c -> b t", reduction="mean"),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            post_processing_pipeline_embedding = [
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t t1)", t1=chunk_num),
                # dict(type="Reduce", keys=["feats"], ops="b t c -> b t", reduction="mean"),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
        ),
    ),
    type="ActionFormer",
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=768,
        max_seq_len=window_size,
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        out_channels=512,
        arch=(2, 2, 5),  # layers in embed / stem / branch
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        path_pdrop=0.1,
        use_abs_pe=False,
        freeze_detector=True,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
    )

model = dict(backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="/tmp/tad/OpenTAD/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth",   #"/tmp/tad/AttackTAD/pretrained/new_epoch_6.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                # dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b t (t1 c)", t1=chunk_num),
                # dict(type="Reduce", keys=["feats"], ops="b t c -> b t", reduction="mean"),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            post_processing_pipeline_embedding = [
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t t1)", t1=chunk_num),
                # dict(type="Reduce", keys=["feats"], ops="b t c -> b t", reduction="mean"),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
        ),
    ),
    type="ActionFormer",
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=768,
        max_seq_len=window_size,
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        out_channels=512,
        arch=(2, 2, 5),  # layers in embed / stem / branch
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        path_pdrop=0.1,
        use_abs_pe=False,
        freeze_detector=True,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)


scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=100)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

work_dir = "exps/thumos/attacktad/transfer_adatad"
