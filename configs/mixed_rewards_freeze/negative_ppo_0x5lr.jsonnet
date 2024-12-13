(import 'rho_ppo_gsm8k_freeze.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'none',
            negative_loss_method: 'ppo',
            relative_lr_ppo: 0.5,
        },
    },
}