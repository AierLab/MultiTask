default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        by_epoch=False,
        max_keep_ckpts=90,
        save_best='Out_PSNR',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
dataset_type = 'WSG_RAIN_TRAIN'
dataset_eval_type = 'WSG_OUTDOORRAIN'
max_iters = 40000
train_dataloader = dict(
    batch_size=16,
    num_workers=64,
    dataset=dict(
        type='WSG_RAIN_TRAIN',
        txt_path='/home/4paradigm/set1/data_txt/train/rain.txt',
        root_path='/home/4paradigm/set1/rain/train/',
        crop_size=224,
        fix_sample=9000,
        regular_aug=True,
        test_mode=False))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    drop_last=False,
    dataset=dict(
        type='WSG_OUTDOORRAIN',
        txt_path='/home/4paradigm/set1/data_txt/test/raintest1.txt',
        root_path='/home/4paradigm/set1/rain/train/',
        fix_sample=500,
        test_mode=True))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    drop_last=False,
    dataset=dict(
        type='WSG_OUTDOORRAIN',
        txt_path='/home/4paradigm/set1/data_txt/test/raintest1.txt',
        root_path='/home/4paradigm/set1/rain/train/',
        fix_sample=500,
        test_mode=True))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0006, betas=(
            0.9,
            0.999,
        ), weight_decay=0.02),
    clip_grad=None)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        begin=50,
        end=40000,
        T_max=40000,
        by_epoch=False),
]
val_evaluator = dict(
    type='WSGMetric', metric=[
        'Out_PSNR',
    ])
test_evaluator = dict(
    type='WSGMetric', metric=[
        'Out_PSNR',
    ])
model = dict(
    type='WSG_Model', backbone=dict(type='UNet', base_channel=18, num_res=6))
work_dir = './mnt/pipeline_1/MLT'
launcher = 'pytorch'
