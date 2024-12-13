(import '../polIter_rho1bSft2_vineppo_GSM8K.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'none',
            negative_loss_method: 'ppo',
            relative_lr_ppo: 0.5,
        },
    },
}