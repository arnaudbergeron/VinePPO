(import 'rho_ppo_gsm8k_freeze.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'sppo',
            negative_loss_method: 'ppo',
            relative_lr_sppo: 1.5,
        },
    },
}