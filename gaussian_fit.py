import astropy.nddata.utils as utils
from lineplots import GetNthColumn, GetNthRow
import numpy as np
import sympy as sp
from sympy.abc import A,a,x,y,c,b
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import  griddata

from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import models, fitting
import warnings

import lmfit
from lmfit.lineshapes import  gaussian2d, lorentzian

basePath = '/media/al-chromebook/USB20FD/Python/Research/fits/Full Maps/'
# basePath = '/media/al-linux/USB20FD/Python/Research/fits/Full Maps/'
sofia = 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
sofia_full = basePath + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
hdu = fits.open(sofia_full)[0]
sofia_full_image = hdu.data

import time
import os.path

import json


new_dict = {}


def MoffatFit(coords):
    # inner_dict['file'] = sofia
    # inner_dict[f'{coords}']['file'] = sofia
    # new_dict[f'{coords}'] = {'file': sofia}

    from astropy.nddata import Cutout2D 
    size = 30
    x,y = np.mgrid[:size, :size]
    data = sofia_full_image
    Data = Cutout2D(data, (coords), size).data

    p_init = models.Moffat2D(x_0=size / 2, y_0=size / 2,amplitude=np.nanmax(Data) )
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, Data)

    fwhm = p.fwhm*.768
    print(f'{coords=} and {fwhm=}')
    new_dict['fwhm'] = fwhm
    new_dict[f'amplitude'] = f'{ p.amplitude }'

    # Plot the Data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(Data, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(Data - ( p(x, y)*1.125 ), origin='lower', interpolation='nearest')
    plt.title("Residual")

    # basePath = '/media/al-chromebook/USB20FD/MIRA-CLI/Figures/'
    # basePath = '/media/al-linux/USB20FD/MIRA-CLI/Figures/'
    # newPath = f'{basePath}{sofia}/'
    # if os.path.isdir(newPath) == False:
    #     os.mkdir(newPath)
    # newFile = time.strftime('%Y%m%d-%H%M%S')
    # newFile = f'{newPath}{newFile}_X{coords[0]}_Y{coords[1]}.png'
    # plt.savefig(newFile)
    # json_dict[f'{ coords }'].add(inner_dict)

    ## save the data to a json
    # with open(newPath + 'data.json', 'r+') as file:
        # file_data = json.load(file)

        # file_data.update({f'{coords}': new_dict})
        # file.seek(0)
        # json.dump(file_data, file, indent=4)

    plt.show()
    

# MoffatFit((1145, 3415))