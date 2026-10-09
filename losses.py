from typing import Optional
import torch
import torch.nn.functional as F
from .inputs import GraphInputs

def prototype_info_nce(anchor, centers, labels, temperature):
    logits = F.normalize(anchor, dim=-1) @ F.normalize(centers.detach(), dim=-1).T
    return F.cross_entropy(logits / temperature, labels)


def relation_consistency_loss(long_interest, short_interest, long_prototype,
                              short_prototype, delta=0.1, margin=0.0, beta=1.0):
    relation = F.cosine_similarity(long_prototype.detach(), short_prototype.detach(), dim=-1)
    relation = relation.clamp(-1, 1).detach()
    similarity = F.cosine_similarity(long_interest, short_interest, dim=-1).clamp(-1, 1)
    align = F.relu(relation - delta) / (1 - delta)
    decorrelate = F.relu(-relation - delta) / (1 - delta)
    return (align * (1 - similarity) + beta * decorrelate * F.relu(similarity - margin).square()).mean()


def link_prediction_loss(nodes, pos, neg, predictor):
    if pos is None or neg is None or pos.numel() == 0 or neg.numel() == 0:
        raise ValueError('provide nonempty positive and negative link samples for each graph branch')
    pairs = torch.cat((pos, neg), dim=1)
    logits = predictor(torch.cat((nodes[pairs[0]], nodes[pairs[1]]), dim=-1)).squeeze(-1)
    targets = torch.cat((logits.new_ones(pos.size(1)), logits.new_zeros(neg.size(1))))
    return F.binary_cross_entropy_with_logits(logits, targets)


def joint_loss(model, batch, graphs: Optional[GraphInputs] = None, return_components=False):
    cfg = model.config
    users, targets = batch['user_id'], batch['pos_item_id']
    if targets.shape != users.shape or (targets <= 0).any() or (targets >= cfg.num_items).any():
        raise ValueError('pos_item_id must be real item IDs with shape [B]')
    out = model(users, batch['item_seq'], batch['item_seq_len'], return_details=True)

    logits = out['fused'] @ model.item_embedding.weight[1:].T
    losses = {'recommendation': F.cross_entropy(logits, targets - 1)}
    zero = out['fused'].new_zeros(())
    losses.update({key: zero for key in ('long_link', 'short_link', 'long_cl', 'short_cl', 'consistency', 'l2')})
    if cfg.lambda_link:
        if graphs is None:
            raise ValueError('GraphInputs is required when lambda_link > 0')
        long_nodes, short_nodes = model.encode_graphs(graphs)
        losses['long_link'] = link_prediction_loss(long_nodes, graphs.long_pos_edges, graphs.long_neg_edges,
                                             model.long_link_mlp)
        losses['short_link'] = link_prediction_loss(short_nodes, graphs.short_pos_edges, graphs.short_neg_edges,
                                              model.short_link_mlp)
    if cfg.lambda_cl or cfg.lambda_consistency:
        if not bool(model.prototypes_ready):
            raise RuntimeError('call refresh_prototypes(training_history_batches, graphs) before auxiliary training')
        long_labels, short_labels = model.long_labels[users], model.short_labels[users]
        if (long_labels < 0).any() or (short_labels < 0).any():
            raise ValueError('batch contains users absent from the last training-only prototype refresh')
        if cfg.lambda_cl:
            losses['long_cl'] = prototype_info_nce(out['long_interest'], model.long_centers,
                                                   long_labels, cfg.long_temperature)
            losses['short_cl'] = prototype_info_nce(out['short_interest'], model.short_centers,
                                                    short_labels, cfg.short_temperature)
        if cfg.lambda_consistency:
            losses['consistency'] = relation_consistency_loss(
                out['long_interest'], out['short_interest'], model.long_prototypes[long_labels],
                model.short_prototypes[short_labels], cfg.consistency_delta, cfg.consistency_margin,
                cfg.decorrelation_beta)
    if cfg.lambda_l2:
        losses['l2'] = sum(p.square().sum() for p in model.parameters() if p.requires_grad)
    total = (losses['recommendation'] + cfg.lambda_link * (losses['long_link'] + losses['short_link']) +
        cfg.lambda_cl * (losses['long_cl'] + losses['short_cl']) +
        cfg.lambda_consistency * losses['consistency'] + cfg.lambda_l2 * losses['l2'])
    return (total, losses) if return_components else total