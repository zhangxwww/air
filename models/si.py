import torch


buffer = {}

def init_si(model):
    global buffer

    W = {}
    p_old = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            W[n] = p.data.clone().zero_()
            p_old[n] = p.data.clone()
            buffer['{}_SI_prev_task'.format(n)] = p.data.clone()

    return W, p_old

def update_si(model, W, p_old):
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            if p.grad is not None:
                W[n].add_(-p.grad * (p.detach() - p_old[n]))
            p_old[n] = p.detach().clone()

def update_omega(model, W, epsilon=0.1):
    global buffer

    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')

            p_prev = buffer['{}_SI_prev_task'.format(n)]
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            omega_add = W[n] / (p_change ** 2 + epsilon)
            try:
                omega = buffer['{}_SI_omega'.format(n)]
            except KeyError:
                omega = p.detach().clone().zero_()
            omega_new = omega_add + omega

            buffer['{}_SI_prev_task'.format(n)] = p_current
            buffer['{}_SI_omega'.format(n)] = omega_new

def si_loss(model):
    global buffer
    try:
        losses = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                prev_values = buffer['{}_SI_prev_task'.format(n)]
                omega = buffer['{}_SI_omega'.format(n)]
                losses.append((omega * (p - prev_values) ** 2).sum())
        return sum(losses)
    except KeyError:
        return torch.tensor(0, device=next(model.parameters()).device)
