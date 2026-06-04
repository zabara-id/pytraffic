import numpy as np

class MBRP:
    """
    BPR стоимость проезда по ребру
    
    cap - "ёмкость" ребра
    t0 - время свободного проезда (free flow time)
    alpha, beta - погоняемые коэффициенты, обычно (0.15 и 4)
    """
    def __init__(self, cap, t0, alpha=0.15, beta=4, gamma=1):
        self.cap = np.maximum(cap, 1e-12)
        self.t0 = t0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, flow, f_hat, experiment_mask):
        # Расчет самой функции стоимости на рёбрах
        x = flow / self.cap
        brp_values = self.t0 * (1.0 + self.alpha * x**self.beta)
        return brp_values * np.exp(experiment_mask * self.gamma * (flow - f_hat))
