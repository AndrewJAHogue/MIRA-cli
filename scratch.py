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

basePath = '/media/al-chromebook/USB20FD/Python/Research/fits/Full Maps/'
# basePath = '/media/al-linux/USB20FD/Python/Research/fits/Full Maps/'
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
    plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(z - p(x, y), origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
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
    print(x.shape)

    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)
    print(X.shape)

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

    column_data = GetNthColumn(sofia_full_image, 3115)[1] * 10
    row_data = GetNthRow(sofia_full_image, 1145)[1] * 10
    x, y = row_data[1100:1200], column_data[3400:3500]
    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0
    z = gaussian2d(x, y, amplitude=np.nanmax(y), centerx=3, centery=-.5, sigmax=.6, sigmay=.8)
    z += 2*(np.random.rand(*z.shape)-.5)
    error = np.sqrt(z+1)

    # X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(np.nanmin(y), np.nanmax(y), 100))
    # Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

    # fig, ax = plt.subplots()
    # art = ax.pcolor(X, Y, Z, shading='auto')
    # plt.colorbar(art, ax=ax, label='z')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.show()

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, x, y)
    result = model.fit(z, x=x, y=y, params=params, weights=1/error)
    lmfit.report_fit(result)

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
    # y, x = np.mgrid[0:100, 0:100]
    x = GetNthColumn(sofia_full_image, 3115)[1]
    y = GetNthRow(sofia_full_image, 1145)[1]
    x = x[:len(y)]
    # data = sofia_full_image[3400:3500, 1100:1200]

    z = gaussian2d(x, y, amplitude=1)
    z += 2*(np.random.rand(*z.shape)-.5)
    error = np.sqrt(z+1)

    # X = np.arange(0, data.shape[1], 1) #Stars at 0, increases by 1, goes to length of axis
    # Y = np.arange(0, data.shape[0], 1) #Stars at 0, increases by 1, goes to length of axis
    # # xx, yy = np.meshgrid(x, y) #creates a grid to plot the function over
    # Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, x, y)
    result = model.fit(z, x=x, y=y, params=params, weights=1/error)
    lmfit.report_fit(result)

def attempt_three():
    column_data = GetNthColumn(sofia_full_image, 1145)
    x, y = column_data[0], column_data[1]

    # Fit the data using a Gaussian
    g_init = models.Gaussian2D()
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y)

def setup_2dData():
    data = sofia_full_image[3400:3500, 1100:1200]

    plt.pcolor(data)
    plt.show()

def example_2d_image():
    ## courtesy of https://stackoverflow.com/questions/50559569/how-can-i-make-my-2d-gaussian-fit-to-my-image

    import numpy as np
    import astropy.io.fits as fits
    import os
    from astropy.stats import mad_std
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from lmfit.models import GaussianModel
    from astropy.modeling import models, fitting

    def gaussian(xycoor,x0, y0, sigma, amp):
        '''This Function is the Gaussian Function'''

        x, y = xycoor # x and y taken from fit function.  Stars at 0, increases by 1, goes to length of axis
        A = 1 / (2*sigma**2)
        eq =  amp*np.exp(-A*((x-x0)**2 + (y-y0)**2)) #Gaussian
        return eq


    def fit(image):
        med = np.median(image)
        image = image-med
        # image = image[0,0,:,:]

        max_index = np.where(image >= np.max(image))
        x0 = max_index[1] #Middle of X axis
        y0 = max_index[0] #Middle of Y axis
        x = np.arange(0, image.shape[1], 1) #Stars at 0, increases by 1, goes to length of axis
        y = np.arange(0, image.shape[0], 1) #Stars at 0, increases by 1, goes to length of axis
        xx, yy = np.meshgrid(x, y) #creates a grid to plot the function over
        sigma = np.std(image) #The standard dev given in the Gaussian
        amp = np.max(image) #amplitude
        guess = [x0, y0, sigma, amp] #The initial guess for the gaussian fitting

        low = [0,0,0,0] #start of data array
        #Upper Bounds x0: length of x axis, y0: length of y axis, st dev: max value in image, amplitude: 2x the max value
        upper = [image.shape[0], image.shape[1], np.max(image), np.max(image)*2] 
        bounds = [low, upper]

        params, pcov = curve_fit(gaussian, (xx.ravel(), yy.ravel()), image.ravel(),p0 = guess, bounds = bounds) #optimal fit.  Not sure what pcov is. 

        return params


    def plotting(image, params):
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.scatter(params[0], params[1],s = 10, c = 'red', marker = 'x')
        circle = Circle((params[0], params[1]), params[2], facecolor = 'none', edgecolor = 'red', linewidth = 1)

        ax.add_patch(circle)
        plt.show()

    data = sofia_full_image[5400:5440, 320:360]
    med = np.median(data) 
    data = data - med

    # data = data[0,0,:,:]

    parameters = fit(data)
    xcenter, ycenter, sigmaX, sigmaY = parameters[0], parameters[1], parameters[2], parameters[3]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    print(parameters)
    print(FWHM_x)
    print(FWHM_y)

    #generates a gaussian based on the parameters given
    plotting(data, parameters)

def fwhm():
    # https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7

    def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
        # """Function to fit, returns 2D gaussian function as 1D array"""
        x, y = xy
        xo = float(xo)
        yo = float(yo)    
        g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
        return g.ravel()

    def getFWHM_GaussianFitScaledAmp(img):
        import scipy.optimize as opt
        """Get FWHM(x,y) of a blob by 2D gaussian fitting
        Parameter:
            img - image as numpy array
        Returns: 
            FWHMs in pixels, along x and y axes.
        """
        img[np.isnan(img)] = 0
        img[np.isinf(img)] = 0
        x = np.linspace(0, img.shape[1], img.shape[1])
        y = np.linspace(0, img.shape[0], img.shape[0])
        x, y = np.meshgrid(x, y)
        xy = x,y
        #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
        initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
        # subtract background and rescale image into [0,1], with floor clipping
        bg = np.percentile(img,5)
        img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
        img_scaled[np.isnan(img_scaled)] = 0
        popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, xy, img_scaled.ravel(), p0=initial_guess, bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1), (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
        xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
        FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
        return (FWHM_x * 2, FWHM_y * 2)


    def peak(x, c):
        return np.exp(-np.power(x - c, 2) / 16.0)

    def lin_interp(x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(x, y):
        half = max(y)/2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [lin_interp(x, y, zero_crossings_i[0], half), lin_interp(x, y, zero_crossings_i[1], half)]

    data = sofia_full_image[3100:3150, 290:340]
    print(half_max_x(data[0], data[1]))

def method_poly():
    from astropy.nddata import Cutout2D 
    size = 30
    x,y = np.mgrid[:size, :size]
    data = sofia_full_image
    Data = Cutout2D(data, (1145, 3415), size).data

    # p_init = models.Polynomial2D(degree=20)
    # p_init = models.Gaussian2D(amplitude=1.)
    p_init = models.Moffat2D(x_0=15, y_0=15)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, Data)

    print(p.fwhm*.768)

    # Plot the Data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(Data, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(Data - p(x, y), origin='lower', interpolation='nearest')
    plt.title("Residual")

    plt.show()


method_poly()