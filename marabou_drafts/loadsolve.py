from maraboupy import Marabou
from maraboupy import MarabouCore

query = MarabouCore.loadQuery('abv-query.txt')
options = Marabou.createOptions(verbosity=2, tighteningStrategy='none')
status, vars, stats = MarabouCore.solve(query, options)

print(status, vars, stats)
