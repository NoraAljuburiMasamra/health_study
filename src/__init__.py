
import numpy as np
from vis import plot_hist

class HealthAnalyzer:
    def __init__(self, data):
        self.data = data

    def compute_confidence_interval(self, bootstrap_diffs):
        """Beräknar 95% konfidensintervall."""
        ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
        return ci_low, ci_high

    def plot_distribution(self, bootstrap_diffs):
        """Ritar histogram för bootstrap-fördelning."""
        plot_hist(bootstrap_diffs, title="Bootstrap Distribution", xlabel="Difference")

