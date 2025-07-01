import logging
from pyomo.environ import *
logger = logging.getLogger(__name__)

def create_model(T_val: int, I_max_val: int):
    """
    Mathematical formulation for the sampling allocation problem.

    Args:
        T_val (int): Number of time periods (>= 1).
        I_max_val (int): Number of sites (>= 2).

    Returns:
        ConcreteModel: A model with all parameters, sets, variables, constraints, and objective.

    Raises:
        ValueError: If T_val < 1 or I_max_val < 2.
        Exception: Logs creation errors to 'log.txt' and re-raises.
    """
    # Validate inputs
    if not isinstance(T_val, int) or T_val < 1:
        raise ValueError(f"T_val must be an integer >= 1, got {T_val}")
    if not isinstance(I_max_val, int) or I_max_val < 2:
        raise ValueError(f"I_max_val must be an integer >= 2, got {I_max_val}")

    try:
        model = ConcreteModel()

        # --- Scalar parameters ---
        model.T = Param(mutable=True, initialize=T_val)
        model.I_max = Param(mutable=True, initialize=I_max_val)
        model.N_max = Param(mutable=True, initialize=3)

        # --- Sets ---
        model.Time = Set(initialize=lambda m: range(0, int(value(m.T)) + 1))
        model.Time_noT = Set(initialize=lambda m: range(0, int(value(m.T))))
        model.I = RangeSet(1, I_max_val)

        # --- Other scalar parameters ---
        model.p1 = Param(mutable=True, initialize=0.005)
        model.p2 = Param(mutable=True, initialize=0.01)
        model.cv = Param(mutable=True, initialize=100)
        model.budget = Param(mutable=True, initialize=450000)
        model.alpha0 = Param(mutable=True, initialize=0.05)
        model.alpha1 = Param(mutable=True, initialize=0.01)
        model.c_1 = Param(mutable=True, initialize=1)
        model.c_2 = Param(mutable=True, initialize=1)

        # --- Indexed parameters ---
        model.cu = Param(model.I, mutable=True)
        model.init_P0 = Param(model.I, mutable=True)
        model.init_P1 = Param(model.I, mutable=True)
        model.init_P2 = Param(model.I, mutable=True)
        model.lambda_val = Param(model.I, mutable=True)
        model.theta = Param(model.I, mutable=True)
        model.theta1 = Param(model.I, mutable=True)
        model.p1_e = Param(model.I, mutable=True)
        model.p2_e = Param(model.I, mutable=True)

        # --- Parameter indexed by (Time, I) ---
        model.v = Param(model.Time, model.I, mutable=True)

        # --- Decision Variables ---
        model.u = Var(model.Time, model.I, bounds=(0, 1))
        model.P0 = Var(model.Time, model.I, bounds=(0, 1))
        model.P1 = Var(model.Time, model.I, bounds=(0, 1))
        model.P2 = Var(model.Time, model.I, bounds=(0, 1))

        # --- Helper for objective ---
        def prod_over_other_sites(m, t, i):
            prod_val = 1
            for j in m.I:
                if j != i:
                    prod_val *= (m.P0[t, j] + m.P1[t, j] + m.P2[t, j])
            return prod_val

        # --- Objective ---
        def objective_rule(m):
            Tn = int(value(m.T))
            term1 = sum(
                (m.c_1 * m.P1[t, i] + m.c_2 * m.P2[t, i]) * prod_over_other_sites(m, t, i)
                for t in m.Time_noT for i in m.I
            )
            term2 = sum(
                (m.p1_e[i] * m.P1[Tn, i] + m.p2_e[i] * m.P2[Tn, i]) * prod_over_other_sites(m, Tn, i)
                for i in m.I
            )
            return term1 + term2
        model.J = Objective(rule=objective_rule, sense=minimize)

        # --- Constraints ---
        model.CS_state0 = Constraint(
            model.Time_noT, model.I,
            rule=lambda m, t, i: m.P0[t+1, i]
                == m.P0[t, i] - m.theta[i] * exp(-m.alpha0 * m.v[t, i]) * m.P0[t, i]
        )
        model.CS_state0_init = Constraint(
            model.I,
            rule=lambda m, i: m.P0[0, i] == m.init_P0[i]
        )

        model.CS_state1 = Constraint(
            model.Time_noT, model.I,
            rule=lambda m, t, i: (
                m.P1[t+1, i] == m.P1[t, i]
                + m.theta[i] * exp(-m.alpha0 * m.v[t, i]) * m.P0[t, i]
                - m.theta1[i] * exp(-m.alpha1 * m.v[t, i]) * m.P1[t, i]
                - (1 - exp(-m.lambda_val[i] * m.p1 * m.u[t, i])) * (
                    m.theta[i] * exp(-m.alpha0 * m.v[t, i]) * m.P0[t, i]
                    + (1 - m.theta1[i] * exp(-m.alpha1 * m.v[t, i])) * m.P1[t, i]
                )
            )
        )
        model.CS_state1_init = Constraint(
            model.I,
            rule=lambda m, i: m.P1[0, i] == m.init_P1[i]
        )

        model.CS_state2 = Constraint(
            model.Time_noT, model.I,
            rule=lambda m, t, i: (
                m.P2[t+1, i] == m.P2[t, i]
                + m.theta1[i] * exp(-m.alpha1 * m.v[t, i]) * m.P1[t, i]
                - (1 - exp(-m.lambda_val[i] * m.p2 * m.u[t, i])) * (
                    m.theta1[i] * exp(-m.alpha1 * m.v[t, i]) * m.P1[t, i] + m.P2[t, i]
                )
            )
        )
        model.CS_state2_init = Constraint(
            model.I,
            rule=lambda m, i: m.P2[0, i] == m.init_P2[i]
        )

        model.CS_budget = Constraint(
            model.Time,
            rule=lambda m, t: sum(
                m.cu[i] * m.lambda_val[i] * m.u[t, i] + m.cv * m.v[t, i]
                for i in m.I
            ) <= m.budget
        )

        return model

    except Exception:
        logger.exception("Error in create_model")
        raise
