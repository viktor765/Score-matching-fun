# %%

import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# checkerboard data
def checkerboard_data(n):
    x1 = np.random.rand(3*n) * 4 - 2
    x2 = np.random.rand(3*n) * 4 - 2
    y = np.zeros(3*n)
    for i in range(3*n):
        #if (np.floor(x1[i]) + np.floor(x2[i])) % 2 == 0:
        if (x1[i] * x2[i]) > 0:
            y[i] = 1
        else:
            y[i] = 0

    I = y == 0
    x1 = x1[I][:n]
    x2 = x2[I][:n]
    y = y[I][:n]

    return x1, x2, y

n = 20000

data = checkerboard_data(n)

x1 = data[0]
x2 = data[1]
y = data[2]

plt.scatter(x1[:1000], x2[:1000], c=y[:1000], s=1)
plt.show()

# %%

train_data = torch.tensor(np.array([x1, x2]).T, dtype=torch.float32, device=device)
#train_labels = torch.tensor(y, dtype=torch.float32)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

# %%

n_epochs = 100
learning_rate = 0.0001
layers = [3, 4000, 4000, 2]

# %%

class ScoreModel(nn.Module):
    def __init__(self, layers):
        super(ScoreModel, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(layers[i+1]))
        
    def forward(self, x):
        for layer, bn in zip(self.layers[:-1], self.batch_norms[:-1]):
            x = layer(x)
            #x = bn(x)
            x = torch.relu(x)

        x = self.layers[-1](x)
        
        return x

# %%

sigma_sq_final = 16

def sigma_sq(t):
    return sigma_sq_final * torch.pow(t, 2)

def sigma_der_sqrt(t):
    return sigma_sq_final**(1/2) * torch.sqrt(2*t)
    
def ve_score(t, x, x_0):
    return -1/sigma_sq(t) * (x - x_0)
    
# %%

model = ScoreModel(layers)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# test
t = 0.1
m = 20
x_sample = train_data[0:m]
print(x_sample)

# stack t and x_sample
sample = torch.cat((t * torch.ones(m,1,device=device), x_sample), 1)
print(sample)
out = model(sample)
print(out)

# %%

losses = []

model.train()
for epoch in range(n_epochs):
    losses_epoch = []
    for i, x_0 in enumerate(train_loader):
        optimizer.zero_grad()

        #t = torch.rand(1, device=device)# TODO: Change to independent for every place?
        #t = t.repeat(x_0.shape[0], 1)
        t = torch.rand(x_0.shape[0], 1, device=device)
        #temp
        #t = 0.05 * torch.ones(x_0.shape[0], 1, device=device)
        sigma_sq_t = sigma_sq(t).to(device)

        x_t = x_0 + torch.sqrt(sigma_sq_t) * torch.randn(x_0.shape, device=device)

        score = ve_score(t, x_t, x_0)

        x_in = torch.cat((t, x_t), 1)
        s = model(x_in)

        loss = torch.mean(sigma_sq_t * (s - score)**2)
        #loss = torch.mean((s - score)**2)
        loss.backward()

        #clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        losses_epoch.append(loss.item())

        #print(f'Loss: {loss.item()}')
    
    losses.append(np.mean(losses_epoch))
    print(f'{epoch=}, mean loss={losses[-1]}')

print(losses)

# %%

plt.plot(np.arange(1, len(losses)+1), losses)
#plt.ylim(0,3)
plt.show()

# %%

model.eval()

t = 0.1

x1 = np.linspace(-4, 4, 21)
x2 = np.linspace(-4, 4, 21)
x1, x2 = np.meshgrid(x1, x2)

x1 = x1.flatten()
x2 = x2.flatten()

x = torch.tensor(np.array([x1, x2]).T, dtype=torch.float32, device=device)

t = t * torch.ones(x.shape[0], 1, device=device)
x_in = torch.cat((t, x), 1).to(device)

s = model(x_in)

s = s.cpu().detach().numpy()

plt.scatter(data[0][:200], data[1][:200], c=data[2][:200], s=1)
plt.scatter(x1, x2, s=1)
plt.quiver(x1, x2, s[:,0], s[:,1])
plt.savefig('checkerboard_learned_score.pdf')
plt.show()

#score = ve_score(t, x, x)
#plt.scatter(x1, x2, s=1)
#plt.quiver(x1, x2, score[:,0], score[:,1])


# %%

device2 = torch.device('cpu')

# move to cpu
model_sim = ScoreModel(layers).to(device)
model_sim.load_state_dict(model.state_dict())
model_sim.to(device2)

# %%

N = 1000
samples = sigma_sq_final**(1/2) * torch.randn((N, 2), device=device2)
dt = 0.01

x_t = samples
for t in np.arange(0, 1, dt)[::-1]:
    x_in = torch.cat((t * torch.ones(x_t.shape[0], 1, device=device2), x_t), 1)
    s = model_sim(x_in)
    g = sigma_der_sqrt(torch.tensor(t))

    dw = torch.sqrt(torch.tensor(dt)) * torch.randn((N, 2), device=device2)

    drift = -g**2 * s
    diffusion_coeff = g

    dx = drift * dt + diffusion_coeff * dw

    x_t = x_t - dx

x_samples_sde = x_t

# %%

plt.scatter(x_samples_sde[:, 0].cpu().detach(), x_samples_sde[:, 1].cpu().detach(), s=1)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.savefig('checkerboard_reverse_sde_generated.pdf')
plt.show()

# %%

N = 1000
samples = sigma_sq_final**(1/2) * torch.randn((N, 2), device=device2)
dt = 0.01

x_t = samples
for t in np.arange(0, 1, dt)[::-1]:
    x_in = torch.cat((t * torch.ones(x_t.shape[0], 1, device=device2), x_t), 1)
    s = model_sim(x_in)
    g = sigma_der_sqrt(torch.tensor(t))

    drift = -1/2 * g**2 * s

    dx = drift * dt

    x_t = x_t - dx

x_samples_ode = x_t

# %%

plt.scatter(x_samples_ode[:, 0].cpu().detach(), x_samples_ode[:, 1].cpu().detach(), s=1)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.savefig('checkerboard_reverse_ode_generated.pdf')
plt.show()

# %%

# plot both
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].scatter(x_samples_sde[:, 0].cpu().detach(), x_samples_sde[:, 1].cpu().detach(), s=1)
ax[0].set_xlim([-5, 5])
ax[0].set_ylim([-5, 5])

ax[1].scatter(x_samples_ode[:, 0].cpu().detach(), x_samples_ode[:, 1].cpu().detach(), s=1)
ax[1].set_xlim([-5, 5])
ax[1].set_ylim([-5, 5])

plt.savefig('checkerboard_reverse_generated.pdf')

plt.show()

# %%

# Compute likelihood

# Cartesian product of x1 and x2
num_points = 201
x1 = np.linspace(-4, 4, num_points)
x2 = np.linspace(-4, 4, num_points)
x1, x2 = np.meshgrid(x1, x2)

x1 = x1.flatten()
x2 = x2.flatten()

x = torch.tensor(np.array([x1, x2]).T, dtype=torch.float32, device=device2)

dt = 0.02

x_t = x
p_t = torch.zeros(x.shape[0], device=device2)
for t in np.arange(0, 1, dt)[::+1]:
    x_in = torch.cat((t * torch.ones(x_t.shape[0], 1, device=device2), x_t), 1).detach().requires_grad_(True)
    #remove gradients
    x_in.grad = None
    model_sim.requires_grad_(False)

    # take divergence of the score
    s = model_sim(x_in)

    g = sigma_der_sqrt(torch.tensor(t))

    drift = -1/2 * g**2 * s

    dx = drift * dt

    # calculate divergence
    div = torch.zeros(x_t.shape[0], device=device2)

    for i in range(2):
        v = torch.zeros_like(s)
        v[:, i] = 1
        x_in.grad = None
        s.backward(v, retain_graph=True)
        div = div + x_in.grad[:, i+1]

    dp = -1/2 * g**2 * div * dt
    p_t = p_t + dp

    x_t = x_t + dx

log_likelihood = -1/2 * x_t.pow(2).sum(1) / sigma_sq_final - 2/2 * np.log(2 * np.pi * sigma_sq_final)
log_likelihood = log_likelihood + p_t
log_likelihood = log_likelihood.cpu().detach().numpy().reshape(num_points, num_points)

x_samples_ode_forward = x_t
# %%
plt.scatter(x_samples_ode_forward[:, 0].cpu().detach(), x_samples_ode_forward[:, 1].cpu().detach(), s=1)
plt.xlim([-26, 26])
plt.ylim([-26, 26])
plt.savefig('checkerboard_likelihood_forward_generated.pdf')
plt.show()
# %%

# plot likelihood

l_ = np.flip(log_likelihood, 0)
#plt.imshow(l_, extent=(-4, 4, -4, 4))
plt.imshow(np.exp(l_).clip(0, 10), extent=(-4, 4, -4, 4))
plt.colorbar()
plt.savefig('checkerboard_reverse_likelihood.pdf')
plt.show()
# %%

plt.hist(x_samples_ode_forward[:, 1].cpu().detach().numpy(), bins=100)
plt.show()

# %%
