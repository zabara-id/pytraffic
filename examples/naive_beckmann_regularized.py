import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTRAFFIC_DIR = ROOT_DIR / "src" / "pytraffic"
sys.path.insert(0, str(PYTRAFFIC_DIR))

import numpy as np
from pytraffic.graph.CSRGraph import CSRGraph
from pytraffic.models.BRPCost import BRP
import pytraffic.models.Beckmann as beckmann

# ============================================================
# Example
# 
# 1) Матрица корреспонденции должна быть с нулевой дагональю
# 2) Граф транпортной сети должен быть связный, иначе можно решить две разные задачи
# 3) Функция стоимости ребра должна зависеть только от загрузки 
# этого ребра, иначе задачу оптимизации будет поставить невозможно 
# 
# ============================================================
if __name__ == "__main__":
    # Количество origin и destination (пока что одинаковое кол-во == кол-ву вершин в графе)
    n_nodes = 4

    # Направленные рёбра (u -> v)
    tail = np.array([0, 1, 0, 2, 1, 2, 1, 3, 2, 3], dtype=np.int32)
    head = np.array([1, 0, 2, 0, 2, 1, 3, 1, 3, 2], dtype=np.int32)
    # Собираем граф
    csr = CSRGraph.from_edges(n_nodes, tail, head)

    # Функция стоимости ребра
    cap = np.array([20, 20, 15, 15, 15, 15, 20, 20, 15, 15], dtype=np.float64)
    t0  = np.array([6, 4, 5, 5, 4, 4, 6, 4, 5, 5], dtype=np.float64)
    alpha = np.full_like(t0, 0.15)
    beta  = np.full_like(t0, 4.0)
    edge_cost = BRP(cap, t0, alpha, beta)

    # OD matrix over ALL nodes (n x n)
    D = np.array([
        [0, 100, 50,  0],
        [80,  0,  20, 10],
        [40, 10,  0, 30],
        [0,  30, 10,  0],
    ], dtype=np.float64)

    # Пусть нам известны какие-то потоки на первых двух рёбрах 
    # (тут в примере важно, чтобы эти потоки не выходили за максимальную корреспонденцию, и наче мы чето странное решаем)

    f_hat = np.zeros_like(cap)
    f_hat[0] = 90
    f_hat[1] = 67
    experiment_mask = np.zeros_like(f_hat)
    experiment_mask[0] = 1
    experiment_mask[1] = 1

    flow, gradient, gradient_for_optimal_potential_value_over_D \
        = beckmann.fw_beckmann_regularized_marginal(csr, edge_cost, D, f_hat, experiment_mask, alpha=0.9, max_iter=5000)

    # А теперь решение классической задачи (равновесное распределение потоков)

    flow_eq, gradient_eq = beckmann.fw_beckmann(csr, edge_cost, D)

    print("Difference between requlirized and equilibriul solution")
    for e in range(csr.m):
        print(f"{tail[e]}->{head[e]}\tflow_r={flow[e]:.3f}\tflow_e={flow_eq[e]:.3f}\tdiff={flow[e]-flow_eq[e]:.3f}") 

    print(gradient_for_optimal_potential_value_over_D.shape)

