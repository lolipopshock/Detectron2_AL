def one_vs_two_scoring(probs):
    """Compute the one_vs_two_scores for the input probabilities


    Args:
        probs (torch.Tensor): NxC tensor
    
    Returns: 
        scores (torch.Tensor): N tensor 
            the one_vs_two_scores
    """

    N, C = probs.shape
    assert C>=2, "the number of classes must be more than 1"

    sorted_probs, _  = probs.sort(dim=-1, descending=True)

    return (1 - (sorted_probs[:, 0] - sorted_probs[:, 1]))
