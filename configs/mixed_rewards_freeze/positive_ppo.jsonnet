(import 'rho_ppo_gsm8k_freeze.jsonnet')

+{
    trainer+: {
        params+: {
            positive_loss_method: 'ppo',
            negative_loss_method: 'none',
        },
    },
}