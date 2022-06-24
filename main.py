import numpy as np
import sympy as sp
from sympy.abc import A,a,x,y,c,b

x0, y0 = sp.symbols('x0, y0')

twoD_G_eq = A*(-(x - x0)** 2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2)

twoD_G_eq