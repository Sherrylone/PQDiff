import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

# x0_pred, noise_pred

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 24, 24)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.train = d(
        n_steps=80000,
        batch_size=256,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=10000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        img_size=24,
        patch_size=2,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1001,
        use_checkpoint=True
    )

    config.dataset = d(
        name='flickr',
        path='./dataset/scenery/train/',
        resolution=192,
        embed_dim=1024,
        grid_size=12,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=50,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    return config
