(import '../polIter_rho1bSft2_vineppo_GSM8K.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'sppo',
            negative_loss_method: 'ppo',
            sppo_clamp_value_low: 1.0,
            sppo_clamp_value_high: 1.0,
            ppo_clamp_value_low: 0.0,
            ppo_clamp_value_high: 1.0,
        },
        general_training_args+: {
            learning_rate: 8e-7
        }
    },
}