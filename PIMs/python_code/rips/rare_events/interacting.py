from rips.utils import *


def quantile(parts: Particles, q):
    return (parts[q - 1].score + parts[q].score) / 2


def interacting_pim(fk: FKModel) -> float:

    # Initialization parameters
    n = fk.num_particles
    s = fk.s
    q = int(n - n / s)
    k = 0

    # Initialize particle trajectories
    parts = fk.sample_particles(n)
    parts = sorted(parts)

    # Starting score
    score = quantile(parts, q)
    fk.add_score(score)

    while score < 1:
        # Sample transitions
        k += 1
        parts = list(fk.sample_transitions(parts[q:], k, score, shuffle=True))

        # New score
        parts = sorted(parts)
        score = quantile(parts, q)
        fk.add_score(score)

    else:
        fk.scores[-1] = 1  # For annealing_pim

    p = sum(part.score >= 1 for part in parts) * s ** - k / n

    return p
