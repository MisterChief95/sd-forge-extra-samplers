import torch

from k_diffusion.sampling import to_d


def clamp(x: int | float, lower: int | float, upper: int | float) -> int | float:
    return max(lower, min(x, upper))


class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask

    def __enter__(self):
        if self.init_latent is not None:
            self.model.init_latent = torch.nn.functional.interpolate(
                input=self.init_latent, size=self.x.shape[2:4], mode=self.mode
            )
        if self.mask is not None:
            self.model.mask = torch.nn.functional.interpolate(
                input=self.mask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode
            ).squeeze(0)
        if self.nmask is not None:
            self.model.nmask = torch.nn.functional.interpolate(
                input=self.nmask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode
            ).squeeze(0)

        return self

    def __exit__(self, type, value, traceback):
        del self.model.init_latent, self.model.mask, self.model.nmask
        self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask


@torch.no_grad()
def overall_sampling_step(x, model, dt, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]

    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **extra_args)
    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = denoised.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]

    x = (
        a_list.view(batch_size, channels, m, n, 2, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch_size, channels, 2 * m, 2 * n)
    )

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x

        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content

        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content

        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]

        x = x_expanded

    return x


@torch.no_grad()
def smea_sampling_step(x, model, dt, sigma_hat, **extra_args):
    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, scale_factor=(1.25, 1.25), mode="nearest-exact")

    with _Rescaler(model, x, "nearest-exact", **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)

    d = to_d(x, sigma_hat, denoised)
    x = x + d * dt
    x = torch.nn.functional.interpolate(input=x, size=(m, n), mode="nearest-exact")

    return x


@torch.no_grad()
def dy_sampling_step(x, model, dt, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with _Rescaler(model, c, "nearest-exact", **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)

    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = (
        a_list.view(batch_size, channels, m, n, 2, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch_size, channels, 2 * m, 2 * n)
    )

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x

        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content

        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content

        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]

        x = x_expanded

    return x


def sampler_metadata(name: str, extra_params: dict = {}):
    def decorator(func):
        func.sampler_extra_params = extra_params
        func.sampler_name = name
        func.sampler_k_name = name.replace(" ", "_").lower()
        return func

    return decorator
