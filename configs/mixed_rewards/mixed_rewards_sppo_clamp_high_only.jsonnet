(import '../polIter_rho1bSft2_ppo_GSM8K.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'sppo',
            negative_loss_method: 'ppo',
            sppo_clamp_value_high: 0.2,
        },
    },
}