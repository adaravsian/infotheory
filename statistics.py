import numpy as np
import scipy

class TestStatistic:
    def __init__(self):
        pass

    def __call__(self, model_path, shuffle=False):
        pass

class BasicStatistic:
    def __init__(self, document_index, metric, n=None, reference_path=None):
        self.texts = document_index.index["texts"]
        self.order = np.asarray(document_index.index["order"])
        self.metric = metric
        self.n = n if n is not None else len(self.texts)
        self.reference_path = reference_path
    
    def __call__(self, model_path, shuffle=False):
        # Restrict to first n samples
        order = self.order[: self.n]
        model_stats = np.asarray(eval_model(model_path, self.texts, self.metric))[: self.n]

        # If a reference model is provided, subtract its stats
        if self.reference_path is not None:
            ref_stats = np.asarray(
                eval_model(self.reference_path, self.texts, self.metric)
            )[: self.n]
            stats_for_corr = model_stats - ref_stats
        else:
            stats_for_corr = model_stats

        # Optional permutation for null tests
        if shuffle:
            perm = np.random.permutation(self.n)
            order = perm[order]

        # Spearman automatically handles ranking
        return scipy.stats.spearmanr(order, stats_for_corr)

def eval_model(model_path, texts, metric):
    return metric(model_path, texts)
