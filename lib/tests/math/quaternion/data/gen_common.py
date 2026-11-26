"""The generator of inputs shared with the tests (`common.rs`), and the format of the data files.

SplitMix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number generators", OOPSLA
2014): the tests generate the same inputs from the same seeds, so that a data file holds only the
outputs of the reference. A uniform float in [0, 1) is the top 53 bits times 2^-53, and every
derived input is a few IEEE operations that Rust and Python round alike.
"""

M = (1 << 64) - 1


class Rng:
    def __init__(self, seed):
        self.state = seed & M

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & M
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) * 2.0**-53

    def signed(self):
        return 2.0 * self.uniform() - 1.0


def line(tag, values):
    """One line of a data file: the tag, then the values, each as the shortest decimal that reads
    back to the same double (Python's repr, which Rust's `parse::<f64>` reads exactly)."""
    return tag + " " + " ".join(repr(float(v)) for v in values)
