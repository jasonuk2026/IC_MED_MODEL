"""
eval_sample_strategy.py

Event sampling strategy used by build_eval_task_data.py.

The strategy function is the single place to change if you want a different
selection policy (e.g. most-recent N, weighted by recency, etc.).
"""

import random


def strategy(event_idxs: list, num_sample: int) -> list:
    """Random sampling strategy for evaluation event selection.

    Args:
        event_idxs : list of valid positions (0-based indices into the filtered
                     event list for a patient).  Only events that are not
                     condition_occurrence and that have a BioLinkBERT embedding
                     are included.
        num_sample : number of events to select.

    Returns:
        Sorted list of selected positions drawn from event_idxs.
        If num_sample >= len(event_idxs), returns all positions unchanged.
    """
    # if num_sample >= len(event_idxs):
    #     return list(event_idxs)
    # return sorted(random.sample(event_idxs, num_sample))

    return event_idxs[-num_sample:]
