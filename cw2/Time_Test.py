import cla_utils
import numpy as np
from numpy import random
from cw2 import Q3
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()
Q3.sim(50, 1, option='Banded')
pr.disable()
pr.dump_stats('example.stats')
p = pstats.Stats('example.stats')

p.print_stats()