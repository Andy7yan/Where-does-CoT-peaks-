"""NLDD helpers for the current experiment."""


def compute_logit_margin(logits: object, gold_token_id: int) -> float:
    """Compute the logit margin for a gold token."""

    raise NotImplementedError("NLDD margin computation is not implemented yet.")


def measure_nldd(clean_margin: float, corrupt_margin: float) -> float:
    """Compute an NLDD score from clean and corrupt margins."""

    raise NotImplementedError("NLDD scoring is not implemented yet.")
