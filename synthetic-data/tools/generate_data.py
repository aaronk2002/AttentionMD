import torch


def data_generator(n, T, d, seed):
    """
    Generate data with n samples, T tokens per sequence, and d dimension
    for each token, with generator seed
    """
    # Use seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Generate randomly
    v = torch.randn((d), generator=generator)
    Y = 2 * torch.ones(n) * (torch.randn(n, generator=generator) > 0) - 1
    X = torch.randn((n, T, d), generator=generator)
    Z = torch.randn((n, d), generator=generator)

    # Normalize
    v /= torch.norm(v)
    X /= torch.norm(X, dim=-1, keepdim=True)
    Z /= torch.norm(Z, dim=-1, keepdim=True)
    return X, Y, Z, v
