# src/_probe_pulp4.py  -- throwaway introspection of PuLP 4.x; delete later.
import sys, inspect, pulp

def show(label, fn):
    try:
        print(f"{label}: {fn()}")
    except Exception as e:
        print(f"{label}: <ERR {type(e).__name__}: {e}>")

print("pulp version:", pulp.__version__)

# (1) how to CREATE a variable in 4.x
show("add_variable sig", lambda: inspect.signature(pulp.LpProblem.add_variable))
for nm in ("add_variable_dict", "add_variable_dicts", "add_variable_matrix"):
    if hasattr(pulp.LpProblem, nm):
        show(f"{nm} sig", lambda nm=nm: inspect.signature(getattr(pulp.LpProblem, nm)))

# (2) the binary category constant
show("LpBinary repr", lambda: repr(pulp.LpBinary))
show("LpCategories repr", lambda: repr(getattr(pulp, "LpCategories", None)))

# (3) confirm the OLD constructor is really dead (caught, so it will not crash)
show("old LpVariable(name,cat=) test",
     lambda: pulp.LpVariable("t", cat=pulp.LpBinary) and "constructed?!")

# (4) how to READ a solved value off the wrapper class
show("LpVariable value-ish attrs",
     lambda: [m for m in dir(pulp.LpVariable) if "val" in m.lower() or m in ("x", "solution_value")])
show("LpVariable has .value?", lambda: hasattr(pulp.LpVariable, "value"))
show("LpVariable.value kind",
     lambda: type(inspect.getattr_static(pulp.LpVariable, "value", None)).__name__)

# (5) best-effort: the underlying rust Variable surface
rc = next((m for n, m in sys.modules.items() if n.endswith("_rustcore")), None) or getattr(pulp, "_rustcore", None)
show("rustcore module", lambda: rc)
if rc is not None and hasattr(rc, "Variable"):
    show("rust Variable value-ish attrs",
         lambda: [m for m in dir(rc.Variable) if "val" in m.lower() or m in ("x", "solution_value")])