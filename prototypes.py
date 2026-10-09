from typing import Iterable, Mapping
import torch
from torch import Tensor
import torch.nn.functional as F
from .inputs import GraphInputs

@torch.no_grad()
def _kmeans(x, k, iterations, restarts, seed):
    x = x.detach().float()
    if k > x.size(0):
        raise ValueError(f'cluster count {k} exceeds training user count {x.size(0)}')
    generator = torch.Generator(device=x.device).manual_seed(seed)

    def assign(centers):
        labels, distances = [], []
        for chunk in x.split(4096):
            distance, label = torch.cdist(chunk, centers).square().min(dim=1)
            labels.append(label)
            distances.append(distance)
        return torch.cat(labels), torch.cat(distances)

    best = None
    for _ in range(restarts):
        centers = x[torch.randperm(x.size(0), generator=generator, device=x.device)[:k]].clone()
        previous_labels = None
        for _ in range(iterations):
            labels, distances = assign(centers)
            if previous_labels is not None and torch.equal(labels, previous_labels):
                break
            previous_labels = labels
            counts = torch.bincount(labels, minlength=k)
            updated = torch.zeros_like(centers).index_add_(0, labels, x)
            updated /= counts.clamp_min(1).unsqueeze(1)
            empty = counts == 0
            if empty.any():

                updated[empty] = x[distances.topk(int(empty.sum())).indices]
            centers = updated
        labels, distances = assign(centers)
        inertia = distances.sum()
        if best is None or inertia < best[0]:
            best = (inertia, centers.clone(), labels.clone())
    return best[1], best[2]


@torch.no_grad()
def refresh_prototypes(model, history_batches: Iterable[Mapping[str, Tensor]], graphs: GraphInputs):
    was_training = model.training
    model.eval()
    try:
        ids, long_rows, short_rows = [], [], []
        for batch in history_batches:
            u = batch['user_id']
            long, short, _ = model.encode_interests(u, batch['item_seq'], batch['item_seq_len'])
            ids.append(u)
            long_rows.append(long)
            short_rows.append(short)
        if not ids:
            raise ValueError('history_batches must contain the training users')
        users = torch.cat(ids)
        if (users <= 0).any() or users.unique().numel() != users.numel():
            raise ValueError('prototype histories need one row per distinct non-padding training user')
        long_nodes, short_nodes = model.encode_graphs(graphs)
        results = []
        for name, nodes, rows, k in (
            ('long', long_nodes, long_rows, model.config.long_clusters),
            ('short', short_nodes, short_rows, model.config.short_clusters),
        ):
            centers, labels = _kmeans(nodes[users], k, model.config.kmeans_iterations,
                                      model.config.kmeans_restarts, model.config.kmeans_seed)
            rows = torch.cat(rows)
            sums = rows.new_zeros(k, rows.size(-1)).index_add_(0, labels, rows)
            counts = torch.bincount(labels, minlength=k).clamp_min(1).unsqueeze(-1)
            prototypes = F.normalize(sums / counts, dim=-1)
            results.append((name, centers, labels, prototypes))

        for name, centers, labels, prototypes in results:
            getattr(model, f'{name}_centers').copy_(centers)
            getattr(model, f'{name}_prototypes').copy_(prototypes)
            lookup = getattr(model, f'{name}_labels')
            lookup.fill_(-1)
            lookup[users] = labels
        model.prototypes_ready.fill_(True)
    finally:
        model.train(was_training)