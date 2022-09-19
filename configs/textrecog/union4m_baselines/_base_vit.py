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
    type='Union4M_SwinT',
    preprocessor=dict(
        type='STN',
        in_channels=3,
        resized_image_size=(32, 64),
        output_image_size=(32, 128),
        num_control_points=20),
    backbone=dict(
        type='VisionTransformer',
        img_size=(32, 128),
        patch_size=(16, 8),
        in_chans=3,
        num_classes=0,
        use_mean_pooling=False),
    decoder=dict(
        type='AttenDecoder',
        max_seq_len=32,
        in_channels=768,
        emb_dims=768,
        attn_dims=768,
        hidden_size=768,
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
