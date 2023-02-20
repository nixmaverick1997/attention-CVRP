import requests
import lkh
from time import monotonic

problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n101-k4.vrp').text
problem = lkh.LKHProblem.parse(problem_str)

solver_path = '/usr/users/bdmagr2/nair/Documents/BDRP_CVRP/attention-CVRP/LKH-3.0.6/LKH'
start_time = monotonic()
rep=lkh.solve(solver_path, problem=problem, max_trials=10000, runs=10)
print(f"For 100 customers: Run time {monotonic() - start_time} seconds")
print(rep)