(import '../polIter_rho1bSft2_ppo_GSM8K.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'ppo',
            negative_loss_method: 'none',
        },
    },
}