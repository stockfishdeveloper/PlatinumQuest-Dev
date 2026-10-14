"""Optional wall-time bound for an owned search; never execute a partial forecast."""
import time


class PlanningBudgetExceeded(RuntimeError):
    pass


def check(engine):
    deadline=getattr(engine,'planning_deadline',None)
    if deadline is not None and time.perf_counter()>=deadline:
        raise PlanningBudgetExceeded('Declared planning wall-time budget exceeded')
