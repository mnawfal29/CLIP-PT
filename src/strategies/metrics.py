from typing import Dict
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Accuracy, Mean
from avalanche.evaluation.metric_results import MetricValue

class WeightedAccuracyPluginMetric(PluginMetric[float]):
    """
    This plugin computes the weighted accuracy across all experiences in a stream
    and emits the result only at the end of the stream (eval mode only).
    """

    def __init__(self):
        super().__init__()
        self._accuracy = Accuracy()
        self._mean = Mean()
        self._experience_samples: Dict[int, int] = {}  # Track samples per experience
        self._experience_accuracies: Dict[int, float] = {}  # Track accuracy per experience

    def reset(self) -> None:
        """Reset the metric state."""
        self._accuracy.reset()
        self._mean.reset()
        self._experience_samples.clear()
        self._experience_accuracies.clear()

    def result(self) -> float:
        """Compute the weighted accuracy across all experiences."""
        return self._mean.result()

    def update(self, strategy):
        """Update the accuracy metric with the current batch."""
        y_true, y_pred = strategy.mb_y, strategy.mb_output
        self._accuracy.update(y_pred, y_true)

    def _package_result(self, strategy) -> MetricValue:
        """Package the result into a MetricValue object."""
        metric_value = self.result()
        metric_name = "SessionAccuracy"
        return MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)
    
    def before_eval(self, strategy):
        """Reset the metric state before evaluation."""
        self.reset()

    def after_eval_iteration(self, strategy):
        """Update the accuracy metric at the end of each evaluation iteration."""
        self.update(strategy)

    def after_eval_exp(self, strategy):
        """Store accuracy and sample count for the current experience."""
        self._mean.update(self._accuracy.result(), len(strategy.experience.dataset))
        self._accuracy.reset()  # Reset accuracy for the next experience

    def after_eval(self, strategy):
        """Compute and emit the weighted accuracy at the end of the stream."""
        return self._package_result(strategy)
