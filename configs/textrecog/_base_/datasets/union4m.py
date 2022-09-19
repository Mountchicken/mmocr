# training sets
union4m_train_root = '../Union4M/training_sets'

union4m_train_hell = dict(
    type='RecogTextDataset',
    data_root=union4m_train_root,
    ann_file='hell.jsonl',
    test_mode=False,
    pipeline=None)

union4m_train_difficult = dict(
    type='RecogTextDataset',
    data_root=union4m_train_root,
    ann_file='difficult.jsonl',
    test_mode=False,
    pipeline=None)

union4m_train_hard = dict(
    type='RecogTextDataset',
    data_root=union4m_train_root,
    ann_file='hard.jsonl',
    test_mode=False,
    pipeline=None)

union4m_train_medium = dict(
    type='RecogTextDataset',
    data_root=union4m_train_root,
    ann_file='medium.jsonl',
    test_mode=False,
    pipeline=None)

union4m_train_simple = dict(
    type='RecogTextDataset',
    data_root=union4m_train_root,
    ann_file='simple.jsonl',
    test_mode=False,
    pipeline=None)

# general test sets
union4m_general_test_root = '../Union4M/test_sets_GN'

union4m_test_gn = dict(
    type='RecogTextDataset',
    data_root=union4m_general_test_root,
    ann_file='annotations.jsonl',
    test_mode=True,
    pipeline=None)

# task specific test sets
union4m_task_test_root = '../Union4M/test_sets_TS'

union4m_test_ts_art = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/art',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_curve = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/curve',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_fcos = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/fcos',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_honest = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/honest',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_honest_ori = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/honest_ori',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_meanless = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/meanless',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_multi = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/multi',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)

union4m_test_ts_multi_oriented = dict(
    type='RecogTextDataset',
    data_root=f'{union4m_task_test_root}/multi_oriented',
    ann_file='annotation.jsonl',
    test_mode=True,
    pipeline=None)
