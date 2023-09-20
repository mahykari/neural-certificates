import sys
from maraboupy import Marabou
from maraboupy import MarabouCore

filename = sys.argv[1]

query = MarabouCore.loadQuery(filename)
options = Marabou.createOptions(
  verbosity=2,
  # tighteningStrategy='none'
)
status, vars, stats = MarabouCore.solve(query, options)

print(status, vars, stats)
