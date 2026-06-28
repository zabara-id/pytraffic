import numpy as np

class MBRP:
    """
    BPR стоимость проезда по ребру
    
    cap - "ёмкость" ребра
    t0 - время свободного проезда (free flow time)
    alpha, beta - погоняемые коэффициенты, обычно (0.15 и 4)
    """
    def __init__(self, f_hat, mask, cap, t0, alpha=0.15, beta=4, gamma=10):
        self.cap = np.maximum(cap, 1e-12)
        self.t0 = t0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mask = mask
        self.f_hat = f_hat

    def __call__(self, flow):
        # Расчет самой функции стоимости на рёбрах
        x = flow / self.cap
        brp_values = self.t0 * (1.0 + self.alpha * x**self.beta)
        return brp_values * np.exp(self.mask * self.gamma * ((flow - self.f_hat) / self.f_hat))
    
    def integral(self, flow):
        # Интеграл при b = 4
        exp_ct = np.exp(self.gamma * flow)
        exp_cx0 = np.exp(self.gamma * self.f_hat)
        
        # Первое слагаемое: (e^{ct} - 1)/c
        term1 = (exp_ct - 1) / self.gamma
        
        # Слагаемые от a * J_4
        term2 = self.alpha * (
            (flow**4 * exp_ct) / self.gamma
            - (4 * flow**3 * exp_ct) / (self.gamma**2)
            + (12 * flow**2 * exp_ct) / (self.gamma**3)
            - (24 * flow * exp_ct) / (self.gamma**4)
            + (24 * (exp_ct - 1)) / (self.gamma**5)
        )
        
        # Умножаем на e^{-c x0}
        I_where_nonzero = (term1 + term2) / exp_cx0
        
        I_where_zero = flow + self.alpha * flow**5 / 5

        return I_where_nonzero * self.mask + I_where_zero * (np.ones_like(self.mask) - self.mask)
