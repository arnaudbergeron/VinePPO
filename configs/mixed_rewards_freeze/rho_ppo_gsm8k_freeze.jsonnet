(import '../polIter_rho1bSft2_ppo_GSM8K.jsonnet')
+ {
    trainer+: {
        critic_ckpt_path: '/network/scratch/a/arnaud.bergeron1/vine_run_sing/experiments/ppo_approx_kl/full_ppo/checkpoints/ckpt--iter_0640--epoch_2.00--step_10240/critic/hf_pretrained',
        disable_critic_training: true
    },
}