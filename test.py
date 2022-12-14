from numpy import log, exp, sin, linspace, random
from lmfit_bo import minimize_bo, Parameters
from pprint import PrettyPrinter


class Residual:
    def __init__(self):
        self.nfev = 0

    def __call__(self, params, x, data, uncertainty):
        amp = exp(params["log_amp"])
        phaseshift = params["phase"]
        freq = params["frequency"]
        decay = exp(params["log_decay"])

        model = amp * sin(x * freq + phaseshift) * exp(-x * x * decay) * x**2
        self.nfev += 1
        return (data - model) / uncertainty


params = Parameters()
params.add("log_amp", value=log(10), min=log(1e-5), max=log(100))
params.add("log_decay", value=log(0.007), min=log(1e-5), max=log(0.1))
params.add("phase", value=0.2, min=0, max=3.14)
params.add("frequency", value=3.0, min=2.5, max=3.5)

# generate synthetic data with noise
x = linspace(1, 100)
noise = random.normal(size=x.size, scale=0.2)
data = 7.5 * sin(x * 0.22 + 2.5) * exp(-x * x * 0.01) * x**2 + noise

# generate experimental uncertainties
uncertainty = abs(0.16 + random.normal(size=x.size, scale=0.05))

res = Residual()
out = minimize_bo(
    res, params, args=(x, data, uncertainty), max_nfev=50, method="bayes_opt"
)
print("method", out.method)
print("aic", out.aic)
print("nfev", out.nfev)
print("nfev check", res.nfev)
print("Parameters")
pp = PrettyPrinter(indent=4)
pp.pprint(out.params.valuesdict())
