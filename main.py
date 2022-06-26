import astropy.nddata.utils as utils
from lineplots import GetNthColumn, GetNthRow
import numpy as np
import sympy as sp
from sympy.abc import A,a,x,y,c,b
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import  griddata

import lmfit
from lmfit.lineshapes import  gaussian2d, lorentzian

# basePath = '/media/al-chromebook/USB20FD/Python/Research/fits/Full Maps/'
basePath = '/media/al-linux/USB20FD/Python/Research/fits/Full Maps/'
sofia_full = basePath + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
hdu = fits.open(sofia_full)[0]
sofia_full_image = hdu.data



from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import models, fitting
import warnings

def PolyFit():
    x,y = np.mgrid[ :128, :128 ]
    z = 2. * x ** 2 - 0.5 * x ** 2 + 1.5 * x * y - 1.
    z += np.random.normal(0., 0.1, z.shape) * 50000.

    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, z)

    # Plot the data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
            vmax=5e4)
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(z - p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
            vmax=5e4)
    plt.title("Residual")

    plt.show()

def gaussFit():
    # Generate fake data
    np.random.seed(0)
    x = np.linspace(-5., 5., 200)
    y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
    y += np.random.normal(0., 0.2, x.shape)

    # Fit the data using a box model.
    # Bounds are not really needed but included here to demonstrate usage.
    t_init = models.Trapezoid1D(amplitude=1., x_0=0., width=1., slope=0.5,
                                bounds={"x_0": (-5., 5.)})
    fit_t = fitting.LevMarLSQFitter()
    t = fit_t(t_init, x, y)

    # Fit the data using a Gaussian
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y)

    # Plot the data with the best-fit model
    plt.figure(figsize=(8,5))
    plt.plot(x, y, 'ko')
    plt.plot(x, t(x), label='Trapezoid')
    plt.plot(x, g(x), label='Gaussian')
    plt.xlabel('Position')
    plt.ylabel('Flux')
    plt.legend(loc=2)
    plt.show()

def base_plot():
    # xy = sofia_full_image[3402:3500, 1100:1200]
    column_data = GetNthColumn(sofia_full_image, 1145)
    # row_data = GetNthRow(sofia_full_image, 1145)[1]
    # cutoff = len(row_data)
    plt.plot(column_data[0], column_data[1])
    plt.show()
    # print()

def attempt_two():
    column_data = GetNthColumn(sofia_full_image, 1145)
    x, y = column_data[0], column_data[1]

    # Fit the data using a Gaussian
    g_init = models.Gaussian1D(amplitude=np.nanmax(y), mean=np.nanmean(y), stddev=np.nanstd(y)*7)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y)

    # Plot the data with the best-fit model
    plt.figure(figsize=(8,5))
    plt.plot(x, y, 'ko')
    # plt.plot(x, t(x), label='Trapezoid')
    plt.plot(x, -g(y) + np.nanmax(y), label='Gaussian')
    plt.xlabel('Position')
    plt.ylabel('Flux')
    plt.legend(loc=2)
    plt.show()

def report_two():
    column_data = GetNthColumn(sofia_full_image, 3115)
    x, y = column_data[0], column_data[1]
    z = gaussian2d(x, y, amplitude=2, centerx=0, centery=0, sigmax=0, sigmay=0)
    error = np.sqrt(z+1)

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, x, y)
    result = model.fit(z, x=x, y=y, params=params, weight=1/error, nan_policy='propagate')
    lmfit.report_fit(result)

def example_limfit():
    npoints = 10000
    np.random.seed(2021)
    x = np.random.rand(npoints)*10 - 4
    y = np.random.rand(npoints)*5 - 3
    z = gaussian2d(x, y, amplitude=30, centerx=2, centery=-.5, sigmax=.6, sigmay=.8)
    z += 2*(np.random.rand(*z.shape)-.5)
    error = np.sqrt(z+1)
    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                    np.linspace(y.min(), y.max(), 100))
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

    fig, ax = plt.subplots()
    art = ax.pcolor(X, Y, Z, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, x, y)
    result = model.fit(z, x=x, y=y, params=params, weights=1/error)
    lmfit.report_fit(result)

    # plot all sets of data
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    vmax = np.nanpercentile(Z, 99.9)

    ax = axs[0, 0]
    art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data')

    ax = axs[0, 1]
    fit = model.func(X, Y, **result.best_values)
    art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Fit')

    ax = axs[1, 0]
    fit = model.func(X, Y, **result.best_values)
    art = ax.pcolor(X, Y, Z-fit, vmin=0, vmax=10, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data - Fit')

    for ax in axs.ravel():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axs[1, 1].remove()
    plt.show()

def method_limfit():
    # data = sofia_full_image[3400:3500, 1100:1200]

    column_data = GetNthColumn(sofia_full_image, 3115)[1]
    row_data = GetNthRow(sofia_full_image, 1145)[1]
    x, y = row_data[1100:1200], column_data[3400:3500]
    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0
    z = gaussian2d(x, y, amplitude=np.nanmax(y), centerx=3, centery=-.5, sigmax=.6, sigmay=.8)
    z += 2*(np.random.rand(*z.shape)-.5)
    error = np.sqrt(z+1)

    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(np.nanmin(y), np.nanmax(y), 100))
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

    fig, ax = plt.subplots()
    art = ax.pcolor(X, Y, Z, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    # model = lmfit.models.Gaussian2dModel()
    # params = model.guess(z, x, y)
    # result = model.fit(z, x=x, y=y, params=params, weights=1/error)
    # lmfit.report_fit(result)

def base_plot2d():
    y, x = np.mgrid[0:500, 0:500]
    data = sofia_full_image[3400:3500, 1100:1200]
    plt.subplot(121)
    plt.imshow(data, origin='lower')


    g_fitt = models.Gaussian2D(np.nanmax(data), np.nanmean(data), np.nanmean(data)*2, 10, 5, theta=0.5)(x, y)
    plt.subplot(122)
    # plt.plot(x, g_fitt)
    plt.imshow(g_fitt, origin='lower')

    plt.show()

def limfit_two():
    y, x = np.mgrid[0:500, 0:500]
    data = sofia_full_image[3400:3500, 1100:1200]
    
    plt.subplot(121)
    plt.imshow(data, origin='lower')

    z = gaussian2d(data)
    plt.subplot(122)
    plt.imshow(z, origin='lower')

    plt.show()

limfit_two()

