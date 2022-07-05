import shutil
import astropy.nddata.utils as utils
from lineplots import GetNthColumn, GetNthRow
import numpy as np
import sympy as sp
from sympy.abc import A,a,x,y,c,b
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import  griddata
import astropy.stats as stats

from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import models, fitting
import warnings

import lmfit
from lmfit.lineshapes import  gaussian2d, lorentzian
import time
import os.path

import json

computer_path = '/media/al-chromebook/'
# computer_path = '/media/al-linux/'


def MoffatFit(coords, file, cutoutSize, saveFigs=False, showPlot = True):
    file_hdu = fits.open(file)[0]
    file_data = file_hdu.data

    new_dict = {}

    filename = spliceFullPath(file) + '_MoffatFit'
    makeNewFolder(filename)
    
    base_figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/'
    figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/{filename}'

    json_file = figPath +  '/data.json'
    # Create the initial json
    if os.path.isfile(json_file) == False:
        shutil.copy2(base_figPath + 'data.json', figPath)

    from astropy.nddata import Cutout2D 
    x,y = np.mgrid[:cutoutSize, :cutoutSize]
    Data = Cutout2D(file_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(Data)
    Data -= median

    p_init = models.Moffat2D(x_0=cutoutSize / 2, y_0=cutoutSize / 2,amplitude=np.nanmax(Data) )
    # p_init = models.Gaussian2D(x_mean=15, y_mean=15)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, Data)

    fwhm = p.fwhm*.768
    # fwhm = p.fwhm *3.2

    print(f'{coords=} and {fwhm=}')
    new_dict['fwhm'] = fwhm
    new_dict[f'amplitude'] = f'{ p.amplitude }'

    # fwhm_x = p.x_fwhm
    # fwhm_y = p.y_fwhm
    # print(f'{fwhm_x=} and {fwhm_y}')
    # new_dict['fwhm_x'] = fwhm_x
    # new_dict['fwhm_y'] = fwhm_y



    # Plot the Data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(Data, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(Data - ( p(x, y)), origin='lower', interpolation='nearest')
    plt.title("Residual")



    if saveFigs:
        SaveFig(filename, coords)

    dumpInfo(filename, new_dict, coords)

    if showPlot:
        plt.show()


def Gauss2D(coords, file, cutoutSize, saveFigs=False, showPlot = True):
    file_hdu = fits.open(file)[0]
    file_data = file_hdu.data

    new_dict = {}

    filename = spliceFullPath(file) + '_Gauss2D'
    makeNewFolder(filename)

    base_figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/'
    figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/{filename}'

    json_file = figPath +  '/data.json'
    # Create the initial json
    if os.path.isfile(json_file) == False:
        shutil.copy2(base_figPath + 'data.json', figPath)

    from astropy.nddata import Cutout2D 
    x,y = np.mgrid[:cutoutSize, :cutoutSize]
    Data = Cutout2D(file_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(Data)
    Data -= median

    # p_init = models.Moffat2D(x_0=size / 2, y_0=size / 2,amplitude=np.nanmax(Data) )
    p_init = models.Gaussian2D(x_mean=15, y_mean=15)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, Data)

    fwhm_x = p.x_fwhm*3.2
    fwhm_y = p.y_fwhm*3.2
    print(f'{fwhm_x=} and {fwhm_y=} and fwhm={np.sqrt(fwhm_x*fwhm_y)}')
    new_dict['fwhm'] = np.sqrt(fwhm_x*fwhm_y)
    new_dict['fwhm_x'] = fwhm_x
    new_dict['fwhm_y'] = fwhm_y



    # Plot the Data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(Data, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(Data - ( p(x, y)), origin='lower', interpolation='nearest')
    plt.title("Residual")



    if saveFigs:
        SaveFig(filename, coords)

    dumpInfo(filename, new_dict, coords)

    if showPlot:
        plt.show()


def makeNewFolder(filename):
    basePath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/'
    newPath = f'{basePath}{filename}/'

    if os.path.isdir(newPath) == False:
        os.mkdir(newPath)

def SaveFig(filename, coords):
    figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/{filename}/'

    # newFile = time.strftime('%Y%m%d-%H%M%S')
    newFile = f'{figPath}_X{coords[0]}_Y{coords[1]}.png'
    plt.savefig(newFile)

def dumpInfo(filename, info, coords):
    base_figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/'
    figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/{filename}'

    json_file = figPath +  '/data.json'
    # Create the initial json
    if os.path.isfile(json_file) == False:
        shutil.copy2(base_figPath + 'data.json', figPath)
    ## save the data to a json
    with open(json_file, 'r+') as f:
        f_data = json.load(f)

        f_data.update({f'{coords}': info})
        f.seek(0)
        json.dump(f_data, f, indent=4)

def spliceFullPath(file):
    index = file.rfind('/') + 1
    return file[index:]


def get_fwhm_gauss_file(coords, file, cutoutSize, saveFigs=False, showPlot = True):
    file_hdu = fits.open(file)[0]
    file_data = file_hdu.data

    new_dict = {}

    from astropy.nddata import Cutout2D 
    x,y = np.mgrid[:cutoutSize, :cutoutSize]
    Data = Cutout2D(file_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(Data)
    Data -= median

    # p_init = models.Moffat2D(x_0=size / 2, y_0=size / 2,amplitude=np.nanmax(Data) )
    p_init = models.Gaussian2D(x_mean=cutoutSize/2, y_mean=cutoutSize/2)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, Data)

    fwhm_x = p.x_fwhm
    fwhm_y = p.y_fwhm
    print(f'{fwhm_x=} and {fwhm_y=} and fwhm={np.sqrt(fwhm_x*fwhm_y)}')
    new_dict['fwhm'] = np.sqrt(fwhm_x*fwhm_y)
    new_dict['fwhm_x'] = fwhm_x
    new_dict['fwhm_y'] = fwhm_y

    return new_dict
    
def get_fwhm_gauss_data(coords, data, cutoutSize, saveFigs=False, showPlot = True):
    file_data = data

    new_dict = {}

    from astropy.nddata import Cutout2D 
    x,y = np.mgrid[:cutoutSize, :cutoutSize]
    Data = Cutout2D(file_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(Data)
    # Data -= median

    # p_init = models.Moffat2D(x_0=size / 2, y_0=size / 2,amplitude=np.nanmax(Data) )
    p_init = models.Gaussian2D(x_mean=cutoutSize / 2, y_mean=cutoutSize / 2)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, Data)

    fwhm_x = p.x_fwhm
    fwhm_y = p.y_fwhm
    print(f'{fwhm_x=} and {fwhm_y=} and fwhm={np.sqrt(fwhm_x*fwhm_y)}')
    new_dict['fwhm'] = np.sqrt(fwhm_x*fwhm_y)
    new_dict['fwhm_x'] = fwhm_x
    new_dict['fwhm_y'] = fwhm_y

    return new_dict
    

# __________________Parameters______________________________________________________________

combinedPath = '/media/al-chromebook/USB20FD/Python/Research/fits/Combined Maps/'
basePath = f'{computer_path}USB20FD/Python/Research/fits/Full Maps/'


# file = 'full_0.0147_2026.fits.fits'
# full_map = computer_path + file 
# spits_hdu = fits.open(full_map)[0]
# full_map_data = spits_hdu.data
# full_map_data = np.nan_to_num(full_map_data, posinf=np.nan)


# sofia = 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
# sofia = basePath + sofia

# ___________________Args______________________________________________________________

# coords = 3115, 298
# size = 30

# ___________________EXECECUTION______________________________________________

# MoffatFit(coords, sofia, size, False, False)


# notes
# compare spits, smear sofia until it matches spits
# then go on to coadding
# 8-10 point sources in spits, ~6 fwhm