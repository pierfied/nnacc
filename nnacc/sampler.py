import torch
from tqdm.auto import tqdm
from .HMCSampler import HMCSampler

class Sampler:
    def __init__(self, lnp, x0=None, m=None, transform=None, device='cpu'):
        self.lnp = lnp
        self.transform = transform
        self.device=device

        if x0 is None:
            self.x0 = torch.randn(self.nparams, device=self.device)
        else:
            self.x0 = x0.to(dtype=torch.float32, device=device)

        self.nparams = len(self.x0)

        if m is None:
            self.m = torch.ones(self.nparams, device=self.device)
        else:
            self.m = m.to(dtype=torch.float32, device=device)

    def calc_hess_mass_mat(self, nsteps=1000, eps=1e-4, resamp_x0=True):
        x = self.x0.clone().requires_grad_()

        pbar = tqdm(range(nsteps))
        for i in pbar:
            lnp = self.lnp(x)
            grad = torch.autograd.grad(lnp, x)[0]
            x = x + grad * eps
            pbar.set_description('log-prob: {}'.format(lnp))

        hess = []
        lnp = self.lnp(x)
        grad = torch.autograd.grad(lnp, x, create_graph=True)[0]
        for i in range(self.nparams):
            hess.append(torch.autograd.grad(grad[i], x, retain_graph=True)[0])
        hess = torch.stack(hess)

        u, m, _ = torch.svd(-hess)
        s = 1 / m

        self.u = u
        self.m = m
        self.orig_lnp = self.lnp
        self.xmap = x.detach().clone()
        self.lnp = lambda x: self.orig_lnp(self.xmap + self.u @ x)
        if self.transform is None:
            self.transform = lambda x: self.xmap + self.u @ x
        else:
            self.orig_transform = self.transform
            self.transform = lambda x: self.orig_transform(self.xmap + self.u @ x)

        if resamp_x0:
            self.x0 = torch.randn(self.nparams, device=self.device) * torch.sqrt(s)

    def sample(self, nburn, burn_steps, burn_eps, nsamp, samp_steps, samp_eps):
        hmc = HMCSampler(self.lnp, self.x0, self.m, self.transform, device=self.device)

        hmc.sample(nburn, burn_steps, burn_eps)
        chain = hmc.sample(nsamp, samp_steps, samp_eps)

        return chain