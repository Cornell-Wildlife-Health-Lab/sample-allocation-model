import logging
from pyomo.environ import *
logger = logging.getLogger(__name__)

def extract_results_pyomo(model: ConcreteModel,verbose: bool = False):
    """
    Solve the Pyomo model with IPOPT, check solver outcome,
    and extract time series results for decision and state variables.

    Args:
        model (ConcreteModel): A fully parameterized Pyomo model.
        verbose (bool): True → show IPOPT iterations; False → silent.

    Returns:
        time: List of period indices [0..T].
        u_values: Dict[i] -> list of u[t,i].
        P0_values: Dict[i] -> list of P0[t,i].
        P1_values: Dict[i] -> list of P1[t,i].
        P2_values: Dict[i] -> list of P2[t,i].
        obj: float optimal objective.

    Raises:
        RuntimeError: If the solver fails to find an optimal solution.
        AttributeError: If required variables are missing from the model.
    """
    # Ensure the decision variable exists
    if not hasattr(model, 'u'):
        err = "Model missing decision variable 'u'; cannot extract results."
        logger.error(err)
        raise AttributeError(err)

    try:
        solver = SolverFactory('ipopt')
        solver.options['max_iter'] = 5000

        # Run the solver
        results = solver.solve(model, tee=verbose)

        # Check solver status
        status = results.solver.status
        term_cond = results.solver.termination_condition
        if status != SolverStatus.ok or term_cond != TerminationCondition.optimal:
            msg = (
                f"IPOPT did not solve optimally "
                f"(status={status}, termination={term_cond})"
            )
            raise RuntimeError(msg)

        # Extract time periods
        T_val = int(value(model.T))
        time = list(range(T_val + 1))

        # Extract time‑series variables
        u_values  = {i: [value(model.u[t, i])  for t in model.Time] for i in model.I}
        P0_values = {i: [value(model.P0[t, i]) for t in model.Time] for i in model.I}
        P1_values = {i: [value(model.P1[t, i]) for t in model.Time] for i in model.I}
        P2_values = {i: [value(model.P2[t, i]) for t in model.Time] for i in model.I}

        # Extract objective
        obj = value(model.J)

        return time, u_values, P0_values, P1_values, P2_values, obj

    except Exception as e:
        logger.error(f"Solver or extraction failed: {e}", exc_info=True)
        raise