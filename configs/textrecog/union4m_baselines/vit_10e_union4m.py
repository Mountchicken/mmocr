_base_ = [
    '../_base_/datasets/union4m.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_base.py',
    '_base_vit.py',
]

train_cfg = dict(max_epochs=10)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', end=1, start_factor=0.001,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', end=20, eta_min=3e-6),
]
# dataset settings
train_list = [
    _base_.union4m_train_hell, _base_.union4m_train_difficult,
    _base_.union4m_train_hard, _base_.union4m_train_medium,
    _base_.union4m_train_simple
]
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test,
    _base_.union4m_test_gn, _base_.union4m_test_ts_art,
    _base_.union4m_test_ts_curve, _base_.union4m_test_ts_fcos,
    _base_.union4m_test_ts_honest, _base_.union4m_test_ts_honest_ori,
    _base_.union4m_test_ts_meanless, _base_.union4m_test_ts_multi,
    _base_.union4m_test_ts_multi_oriented
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=3e-4))

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=256,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(dataset_prefixes=[
    'CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15', 'Union4M_GN',
    'Union4M_TS_ART', 'Union4M_TS_CURVE', 'Union4M_TS_FCOS',
    'Union4M_TS_HONEST', 'Union4M_TS_HONEST_ORI', 'Union4M_TS_MEANLESS',
    'Union4M_TS_MULTI', 'Union4M_TS_MULTI_ORIENTED'
])
test_evaluator = val_evaluator
