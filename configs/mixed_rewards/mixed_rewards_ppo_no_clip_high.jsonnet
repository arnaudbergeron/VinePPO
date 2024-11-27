(import '../polIter_rho1bSft2_ppo_GSM8K.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'sppo',
            negative_loss_method: 'ppo',
            clip_sppo_high: false
        },
    },
}