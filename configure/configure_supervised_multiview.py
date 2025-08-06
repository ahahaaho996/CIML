def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            view=6,
            seed=8,
            n=2386,
            class_num=20,
            training=dict(
                lr=3.0e-4,
                batch_size=256,
                epoch=100,
                lambda1=1e-4,
                lambda2=1e-4,
                beta1=1,
                beta2=0.01,
            ),
            Autoencoder1=dict(
                arch1=[[48, 1024, 1024, 1024, 40], [40, 1024, 1024, 1024, 40], [254, 1024, 1024, 1024, 40],
                       [1984, 1024, 1024, 1024, 40], [512, 1024, 1024, 1024, 40], [928, 1024, 1024, 1024, 40]],
                arch2=[40, 1024, 1024, 1024, 40],
                activation='relu',
                batchnorm=True,
            ),
            Autoencoder2=dict(
                arch1=[[48, 1024, 1024, 1024, 40], [40, 1024, 1024, 1024, 40], [254, 1024, 1024, 1024, 40],
                       [1984, 1024, 1024, 1024, 40], [512, 1024, 1024, 1024, 40], [928, 1024, 1024, 1024, 40]],
                activation='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            view=3,
            seed=8,
            n=4485,
            class_num=15,
            training=dict(
                lr=1.0e-4,
                batch_size=256,
                epoch=150,
                alpha=10,
                lambda2=1e-5,
                lambda1=1e-5,
                beta1=1000,
                beta2=0.001,
            ),
            Autoencoder1=dict(
                arch1=[[20, 1024, 1024, 1024, 512], [59, 1024, 1024, 1024, 512], [40, 1024, 1024, 1024, 512]],
                arch2=[512, 1024, 1024, 1024, 512],
                activation='relu',
                batchnorm=True,
            ),
            Autoencoder2=dict(
                arch1=[[20, 1024, 1024, 1024, 512], [59, 1024, 1024, 1024, 512], [40, 1024, 1024, 1024, 512]],
                arch2=[512, 1024, 1024, 1024, 512],
                activation='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            view=3,
            seed=2,
            n=2100,
            class_num=21,
            training=dict(
                lr=1.0e-4,
                batch_size=256,
                epoch=200,
                alpha=10,
                lambda2=1e-4,
                lambda1=1e-4,
                beta1=100,
                beta2=0.001,
            ),
            Autoencoder1=dict(
                arch1=[[20, 1024, 1024, 1024, 40], [59, 1024, 1024, 1024, 40], [40, 1024, 1024, 1024, 40]],
                arch2=[40, 1024, 1024, 1024, 40],
                activation='relu',
                batchnorm=True,
            ),
            Autoencoder2=dict(
                arch1=[[20, 1024, 1024, 1024, 40], [59, 1024, 1024, 1024, 40], [40, 1024, 1024, 1024, 40]],
                activation='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['NUS']:
        return dict(
            view=6,
            seed=8,
            n=2400,
            class_num=12,
            training=dict(
                lr=3.0e-4,
                batch_size=256,
                epoch=50,
                lambda1=1e-4,
                lambda2=1e-4,
                beta1=1,
                beta2=0.01,
            ),
            Autoencoder1=dict(
                arch1=[[64, 1024, 1024, 1024, 256], [144, 1024, 1024, 1024, 256], [73, 1024, 1024, 1024, 256],
                       [128, 1024, 1024, 1024, 256], [225, 1024, 1024, 1024, 256], [500, 1024, 1024, 1024, 256]],
                arch2=[256, 1024, 1024, 1024, 256],
                activation='relu',
                batchnorm=True,
            ),
            Autoencoder2=dict(
                arch1=[[64, 1024, 1024, 1024, 256], [144, 1024, 1024, 1024, 256], [73, 1024, 1024, 1024, 256],
                       [128, 1024, 1024, 1024, 256], [225, 1024, 1024, 1024, 256], [500, 1024, 1024, 1024, 256]],
                activation='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['MSRC_v1']:
        return dict(
            view=5,
            seed=8,
            n=210,
            class_num=7,
            training=dict(
                lr=3.0e-4,
                batch_size=32,
                epoch=50,
                lambda1=1e-4,
                lambda2=1e-4,
                beta1=1,
                beta2=0.001,
            ),
            Autoencoder1=dict(
                arch1=[[24, 1024, 1024, 1024, 128], [576, 1024, 1024, 1024, 128], [512, 1024, 1024, 1024, 128],
                       [256, 1024, 1024, 1024, 128], [254, 1024, 1024, 1024, 128]],
                arch2=[128, 1024, 1024, 1024, 128],
                activation='relu',
                batchnorm=True,
            ),
            Autoencoder2=dict(
                arch1=[[24, 1024, 1024, 1024, 128], [576, 1024, 1024, 1024, 128], [512, 1024, 1024, 1024, 128],
                       [256, 1024, 1024, 1024, 128], [254, 1024, 1024, 1024, 128]],
                activation='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['NoisyMNIST_select']:
        """The default configs."""
        return dict(
            view=2,
            seed=1,
            n=20000,
            class_num=10,
            training=dict(
                lr=1.0e-4,
                batch_size=256,
                epoch=100,  # 50
                lambda2=1e-4,
                lambda1=1e-4,
                beta1=1,
                beta2=0.01,
            ),
            Autoencoder1=dict(
                arch1=[[784, 1024, 1024, 1024, 40], [784, 1024, 1024, 1024, 40]],
                arch2=[40, 1024, 1024, 1024, 40],
                activation='relu',
                batchnorm=True,
            ),
            Autoencoder2=dict(
                arch1=[[784, 1024, 1024, 1024, 40], [784, 1024, 1024, 1024, 40]],
                activation='relu',
                batchnorm=True,
            ),
        )

    else:
        raise Exception('Undefined data name')
