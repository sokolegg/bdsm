import torch


def minimal_noise_pred_squared_value(
        bdsm_noise_pred,
        **kwargs,
):
    return torch.sum((bdsm_noise_pred) ** 2)


def minimal_noise_pred_abs_value(
        bdsm_latents,
        bdsm_noise_pred,
        t,
        latents,
        **kwargs,
):
    return torch.sum(torch.abs(bdsm_noise_pred))


def euclidean_distance_between_anchor_latents(
        anchor_latents: torch.Tensor,
):
    def _distance(
            bdsm_latents,
            bdsm_noise_pred,
            t,
            latents,
            **kwargs,
    ):
        assert bdsm_latents.shape == anchor_latents.shape, f"bdsm_latents.shape: {bdsm_latents.shape}, " \
                                                             f"anchor_latents.shape: {anchor_latents.shape}"

        return torch.sum((bdsm_latents - anchor_latents) ** 2)

    return _distance

