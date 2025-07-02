import logging
from pyomo.environ import *
logger = logging.getLogger(__name__)


def set_parameters_pyomo(
    model: ConcreteModel,
    T_values: int,
    I_max_val: int,
    budget_val: float,
    cu_dict: dict,
    init_P0_dict: dict,
    init_P1_dict: dict,
    init_P2_dict: dict,
    lambda_dict: dict,
    theta_dict: dict,
    theta1_dict: dict,
    v_dict: dict,
    p1_val: float,
    p2_val: float,
    alpha0_val: float,
    alpha1_val: float,
    p1_e_dict: dict,
    p2_e_dict: dict,
    c1_val: float,
    c2_val: float,
    cv_val: float
):
    """
    Assign values to all Pyomo model parameters before solving.

    Args:
        model (ConcreteModel): The Pyomo model instance.
        T_values (int): Planning horizon.
        I_max_val (int): Number of sites.
        budget_val (float): Total budget.
        cu_dict (dict): Sampling cost per site.
        init_P0_dict (dict): Initial belief P0 per site.
        init_P1_dict (dict): Initial belief P1 per site.
        init_P2_dict (dict): Initial belief P2 per site.
        lambda_dict (dict): Sample rate per site.
        theta_dict (dict): Introduction rate per site.
        theta1_dict (dict): Spread rate per site.
        v_dict (dict): Effort levels keyed by (t, i).
        p1_val (float): Prevalence rate state 1.
        p2_val (float): Prevalence rate state 2.
        alpha0_val (float): Prevention effect.
        alpha1_val (float): Containment effect.
        p1_e_dict (dict): Terminal cost for state 1 per site.
        p2_e_dict (dict): Terminal cost for state 2 per site.
        c1_val (float): Cost/damage state 1.
        c2_val (float): Cost/damage state 2.
        cv_val (float): Sampling cost parameter.

    Raises:
        Exception: Logs and re-raises any errors encountered during assignment.
    """
    try:
        # Scalar parameters
        model.T.set_value(T_values)
        model.I_max.set_value(I_max_val)
        model.budget.set_value(budget_val)
        model.p1.set_value(p1_val)
        model.p2.set_value(p2_val)
        model.alpha0.set_value(alpha0_val)
        model.alpha1.set_value(alpha1_val)
        model.cv.set_value(cv_val)
        model.c_1.set_value(c1_val)
        model.c_2.set_value(c2_val)

        # Site-indexed parameters
        for i in model.I:
            model.cu[i].set_value(cu_dict[i])
            model.init_P0[i].set_value(init_P0_dict[i])
            model.init_P1[i].set_value(init_P1_dict[i])
            model.init_P2[i].set_value(init_P2_dict[i])
            model.lambda_val[i].set_value(lambda_dict[i])
            model.theta[i].set_value(theta_dict[i])
            model.theta1[i].set_value(theta1_dict[i])
            model.p1_e[i].set_value(p1_e_dict[i])
            model.p2_e[i].set_value(p2_e_dict[i])

        # Time-and-site-indexed parameter v
        for t in model.Time:
            for i in model.I:
                model.v[t, i].set_value(v_dict[(t, i)])

    except Exception:
        logger.exception("Error in set_parameters_pyomo")
        raise
