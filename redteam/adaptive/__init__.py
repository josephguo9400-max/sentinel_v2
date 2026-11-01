"""Adaptive attacker harness (milestone adaptive-attacker).

Iteratively crafts indirect prompt injections with an Anthropic model and measures adaptive attack
success rate (ASR) against the served D23 bundle, honestly and within a hard dollar cap. See
docs/design/adaptive-attacker.md for the threat model, goal split and success definitions.

Nothing here prints, logs or persists the API key. The response cache is keyed on request *content*
only (never headers).
"""
