"""Classify the outcome of a Dyck intervention.

On a balanced binary task "accuracy <= 0.03" and "accuracy = 0.5" are different effects:
the first means the model's labels are *inverted*, the second that it predicts one class
for everything. Grouping both as "collapse" mixes them, so we label them separately.
"""

INTACT, PARTIAL, LABEL_FLIP, SINGLE_CLASS = "intact", "partial", "label_flip", "single_class"


def classify(acc, pred_pos_rate=None, base_acc=None, intact_tol=0.05, flip_max=0.10, single_tol=0.05):
    """Return one of intact / partial / label_flip / single_class / chance_ambiguous.

    pred_pos_rate is the fraction of examples predicted positive. Without it (old result
    files), an accuracy near 0.5 can't be told apart from a single-class prediction, so it
    is reported as "chance_ambiguous".
    """
    if pred_pos_rate is not None and (pred_pos_rate <= single_tol or pred_pos_rate >= 1 - single_tol):
        return SINGLE_CLASS
    if acc <= flip_max:
        return LABEL_FLIP
    ref = base_acc if base_acc is not None else 1.0
    if acc >= ref - intact_tol:
        return INTACT
    if pred_pos_rate is None and abs(acc - 0.5) <= 0.02:
        return "chance_ambiguous"
    return PARTIAL
