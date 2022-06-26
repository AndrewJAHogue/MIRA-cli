import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Galactic

def GetNthColumn(file, xvalue, **kwargs):
    xmin = kwargs.get('xmin', None)
    xmax = kwargs.get('xmax', None)

    ymin = kwargs.get('ymin', None)
    ymax = kwargs.get('ymax', None)

    fake = kwargs.get('fake', True)

    yvalues = np.array([])
    yvalues = file[:,xvalue]
    xdata = np.array(range(len(yvalues)))

    if min is not None and max is not None:
        yvalues = yvalues[xmin:xmax]
        xdata = xdata[xmin:xmax]
    # fake x data
    # if fake:
    return np.array([xdata, yvalues])
    # else:
        # return np.array(yvalues)

def GetNthRow(file, yvalue, **kwargs):
    xmin = kwargs.get('xmin', None)
    xmax = kwargs.get('xmax', None)

    ymin = kwargs.get('ymin', None)
    ymax = kwargs.get('ymax', None)

    fake = kwargs.get('fake', True)

    xvalues = np.array([])
    xvalues = file[yvalue]
    ydata = np.array(range(len(xvalues)))

    if min is not None and max is not None:
        xvalues = xvalues[xmin:xmax]
        ydata = ydata[xmin:xmax]
    return np.array([ydata, xvalues])
