"""
Sampling count policy for oversampled EHR timelines.

Edit `get_sample_n_times` to change how many random sub-samples are drawn
for any given (label, timeline length) combination.

Contract:
    - Returns 1 when ACTUAL_NUM_EVENTS <= N_EVENTS_LIMIT  (no oversampling needed)
    - Returns >= 1 otherwise
"""

import math


def get_sample_n_times(
    is_positive: bool,
    N_EVENTS_LIMIT: int,
    ACTUAL_NUM_EVENTS: int,
    task: str = "",
) -> int:
    """Return the number of random sub-samples to draw for one EHR sample.

    Args:
        is_positive:       True if the sample label is positive / non-normal.
        N_EVENTS_LIMIT:    Maximum number of events per timeline window.
        ACTUAL_NUM_EVENTS: Total events available before the prediction time.
        task:              Task name (e.g. "new_pancan", "chexpert"), used to
                           apply task-specific sampling policies.

    Returns:
        Number of times to sample N_EVENTS_LIMIT events from ACTUAL_NUM_EVENTS.
        Always >= 1.
    """
    if ACTUAL_NUM_EVENTS <= N_EVENTS_LIMIT:
        return 1

    if is_positive: return get_pos_sample_n_times(N_EVENTS_LIMIT, ACTUAL_NUM_EVENTS, task=task)
    else: return get_neg_sample_n_times(N_EVENTS_LIMIT, ACTUAL_NUM_EVENTS, task=task)

def get_pos_sample_n_times(
    N_EVENTS_LIMIT: int,
    ACTUAL_NUM_EVENTS: int,
    task: str = "",
):
    base = math.ceil(ACTUAL_NUM_EVENTS / N_EVENTS_LIMIT)
    if task == "new_pancan":
        if base <= 2:
            return base
        elif base <= 6:
            return base * 40
        elif base <= 20:
            return base * 122
        else:
            return base * 200
    
    elif task == "new_hypertension":
        if base <= 2:
            return base
        elif base <= 6:
            return base * 20
        elif base <= 20:
            return base * 40
        else:
            return base * 50
        
    elif task == "new_hyperlipidemia":
        if base <= 2:
            return base
        elif base <= 6:
            return base * 20
        elif base <= 20:
            return base * 20
        else:
            return base * 50
        
    elif task == "new_celiac":
        if base <= 2:
            return base * 4
        elif base <= 6:
            return base * 25
        elif base <= 20:
            return base * 50
        else:
            return base * 55
        
    elif task == "new_lupus":
        if base <= 2:
            return base * 8
        elif base <= 6:
            return base * 50
        elif base <= 20:
            return base * 50
        else:
            return base * 140
        
    elif task == "new_acutemi":
        if base <= 2:
            return base
        elif base <= 6:
            return base * 10
        elif base <= 20:
            return base * 20
        else:
            return base * 75

def get_neg_sample_n_times(
    N_EVENTS_LIMIT: int,
    ACTUAL_NUM_EVENTS: int,
    task: str = "",
):
    base = math.ceil(ACTUAL_NUM_EVENTS / N_EVENTS_LIMIT)
    if task == "new_pancan":
        if base <= 2:
            return 1
        elif base <= 6:
            return base
        elif base <= 20:
            return base * 2
        else:
            return base * 4
    
    elif task == "new_hypertension":
        if base <= 2:
            return 1
        elif base <= 6:
            return base
        elif base <= 20:
            return base * 6
        else:
            return base * 20
    
    elif task == "new_hyperlipidemia":
        if base <= 2:
            return 1
        elif base <= 6:
            return base
        elif base <= 20:
            return base * 2
        else:
            return base * 10
    
    elif task == "new_celiac":
        if base <= 2:
            return 1
        elif base <= 6:
            return base
        elif base <= 20:
            return base
        else:
            return base
    
    elif task == "new_lupus":
        if base <= 2:
            return 1
        elif base <= 6:
            return base
        elif base <= 20:
            return base * 2
        else:
            return base * 2
    
    elif task == "new_acutemi":
        if base <= 2:
            return 1
        elif base <= 6:
            return base
        elif base <= 20:
            return base * 2
        else:
            return base * 4