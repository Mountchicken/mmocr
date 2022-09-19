file_client_args = dict(backend='disk')

dictionary = dict(
    type='Dictionary',
    dict_file=
    '{{ fileDirname }}/../../../dicts/english_digits_symbols_space.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='Union4M_ResNet45',
    preprocessor=dict(
        type='STN',
        in_channels=3,
        resized_image_size=(32, 64),
        output_image_size=(32, 128),
        num_control_points=20),
    backbone=dict(
        type='ResNet',
        in_channels=3,
        stem_channels=[32],
        block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
        arch_layers=[3, 4, 6, 6, 3],
        arch_channels=[32, 64, 128, 256, 512],
        strides=[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)],
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1, layer='BatchNorm2d'),
        ]),
    encoder=None,
    decoder=dict(
        type='AttenDecoder',
        max_seq_len=32,
        in_channels=512,
        emb_dims=512,
        attn_dims=512,
        hidden_size=512,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', flatten=True, ignore_first_char=True),
        dictionary=dictionary,
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=0),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Verticle2Horizontal'),
    dict(type='Resize', scale=(256, 64), keep_ratio=False),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Verticle2Horizontal'),
    dict(type='Resize', scale=(256, 64), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
