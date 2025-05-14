

def pf_rmse_comp_extended(points):
    """
    Identifies the Pareto front from a list of points.
    Each point is a tuple where the first element is RMSE, the second is complexity,
    and any subsequent elements are other metrics.
    Dominance is determined based on RMSE and complexity:
    A point A dominates point B if A's RMSE <= B's RMSE and A's complexity <= B's complexity,
    and at least one of these is strictly smaller.
    The function returns the full original points that are non-dominated.
    """
    pareto = []
    if not points:
        return pareto

    for i, point1 in enumerate(points):
        rmse1 = point1[0]
        comp1 = point1[1]
        dominated = False
        for j, point2 in enumerate(points):
            if i == j:
                continue 

            rmse2 = point2[0]
            comp2 = point2[1]
            if (rmse2 <= rmse1 and comp2 <= comp1) and \
               (rmse2 < rmse1 or comp2 < comp1):
                dominated = True
                break 

        if not dominated:
            pareto.append(point1) 
    pareto.sort(key=lambda x: x[0])
    return pareto
