import numpy as np

class BRP:
    """
    BPR стоимость проезда по ребру
    
    cap - "ёмкость" ребра
    t0 - время свободного проезда (free flow time)
    alpha, beta - погоняемые коэффициенты, обычно (0.15 и 4)
    """
    def __init__(self, cap, t0, alpha=0.15, beta=4):
        self.cap = np.maximum(cap, 1e-12)
        self.t0 = t0
        self.alpha = alpha
        self.beta = beta

    def __call__(self, flow):
        # Расчет самой функции стоимости на рёбрах
        x = flow / self.cap
        x[x < 0] = 0
        return self.t0 * (1.0 + self.alpha * x**self.beta)
    
    def grad(self, flow):
        # Расчет производной (фактически это матрица Якоби, но 
        # по факту она диагональная, так что возвращается просто вектор)
        # Матрицу получаем через np.diag(result)
        x = flow / self.cap
        return (self.t0 * self.alpha * self.beta / self.cap) * x**(self.beta - 1)

    def integral(self, flow):
        return flow + self.alpha * np.pow(flow, self.beta + 1) / self.beta
