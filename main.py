import openpyxl as pe
import gaussian_fit
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# __________________Parameters______________________________________________________________
computer_path = '/media/al-chromebook/'
Combined_path = '/media/al-chromebook/USB20FD/Python/Research/fits/Combined Maps/'
basePath = f'{computer_path}USB20FD/Python/Research/fits/Full Maps/'


# file = 'full_0.0147_2026.fits.fits'
# full_map = computer_path + file 
# spits_hdu = fits.open(full_map)[0]
# full_map_data = spits_hdu.data
# full_map_data = np.nan_to_num(full_map_data, posinf=np.nan)



def csv():
    sofia = 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
    sofia = basePath + sofia

    # basePath = '/media/al-chromebook/USB20FD/'
    # basePath = '/media/al-linux/USB20FD/'
    filename = "SOFIA pointsources"
    data = pe.load_workbook(f'{computer_path}USB20FD/{filename}.xlsx')
    sheet = data.active
    # x_cells, y_cells = sheet['B'][49:52], sheet['C'][49:52]
    x_cells, y_cells = sheet['B'][49:101], sheet['C'][49:101]

    for index_x, x in enumerate(x_cells):
        if 'Corrected' in sheet['I'][index_x + 49].value:
            coords = (x.value, y_cells[index_x].value)
            print(f'{coords=}')
            gaussian_fit.MoffatFit(coords, sofia, 30, True, False)
            gaussian_fit.Gauss2D(coords, sofia, 30, True, False)

def spits_points():
    spits_point_sources = [
        [52,115],
        [35,35],
        [288, 328],
        [258, 337],
        [250, 81],
        [80, 89],
        [90, 309],
        [28, 327]
    ]
            

    size = 30
    baseFits_path = '/media/al-chromebook/USB20FD/Python/Research/fits/'

    sofia = 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
    sofia_path = f'Full Maps/{sofia}' 

    spits_isofield = 'Spitzer24_IsoFields.fits'
    forcast_isofield1= 'Forcast25_isoField1.fits'

    chosen_file = baseFits_path + spits_isofield

    for coords in spits_point_sources:
        # gaussian_fit.MoffatFit(coords, chosen_file, size, True, False)
        gaussian_fit.Gauss2D(coords, chosen_file, size, True, False)


def std_of_fwhm(json_file):
    import json
    all_fwhm = []
    with open(json_file, 'r+') as f:
        f_data = json.load(f)

        for key in f_data:
            all_fwhm.append(f_data[key]['fwhm'])
        
        
    return np.std(all_fwhm)

def avg_fwhm(json_file):
    import json
    all_fwhm = []
    with open(json_file, 'r+') as f:
        f_data = json.load(f)

        for key in f_data:
            all_fwhm.append(f_data[key]['fwhm'])
        
        
    return np.average(all_fwhm)
            
def spits_iso_std():
    base_figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/'

    spits_isofield = 'Spitzer24_IsoFields.fits'

    filename = spits_isofield + '_Gauss2D'
    figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/{filename}'

    json_file = figPath +  '/data.json'
    print(f'{spits_isofield}_stdev = { std_of_fwhm(json_file) }')

    return std_of_fwhm(json_file)

def spits_iso_avg_fwhm():
    base_figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/'

    spits_isofield = 'Spitzer24_IsoFields.fits'

    filename = spits_isofield + '_Gauss2D'
    figPath = f'{computer_path}/USB20FD/MIRA-CLI/Figures/{filename}'

    json_file = figPath +  '/data.json'
    print(f'{spits_isofield}_average = { avg_fwhm(json_file)} +- {std_of_fwhm(json_file)}')

def spits_iso_avg():

    baseFits_path = '/media/al-chromebook/USB20FD/Python/Research/fits/'
    spits_isofield = 'Spitzer24_IsoFields.fits'

    data = fits.open(baseFits_path + spits_isofield)[0].data

    stdev = np.std(data)
    average = np.average(data)
    print(f'{average=} +- {stdev}')

def spits_derivative_map():
    baseFits_path = '/media/al-chromebook/USB20FD/Python/Research/fits/'
    spits_isofield = 'Spitzer24_IsoFields.fits'

    data1 = fits.open(baseFits_path + spits_isofield)[0].data

    spits_point_sources = [
        [52,115],
        [35,35],
        [288, 328],
        [258, 337],
        [250, 81],
        [80, 89],
        [90, 309],
        [28, 327]
    ]

    xvalue, yvalue = 227, 154
    

    column = GetNthColumn(data1, xvalue)
    columnLineplot = plt.subplot(1,2,1)
    plt.title(f'Column-Pixel Saturation at X={xvalue}')
    plt.xlabel('Y Index')
    plt.ylabel('Pixel Value')
    plt.plot(column[0],column[1])

    row = GetNthRow(data1, yvalue)
    rowLineplot = plt.subplot(1,2,2)
    plt.title(f'Row-Pixel Saturation at Y={yvalue}')
    plt.xlabel('X Index')
    plt.ylabel('Pixel Value')
    plt.plot(row[0],row[1])

    data2 = np.diff(data1, 2)

    column = GetNthColumn(data2, xvalue)
    columnLineplot = plt.subplot(1,2,1)
    plt.title(f'Column-Pixel Saturation at X={xvalue}')
    plt.xlabel('Y Index')
    plt.ylabel('Pixel Value')
    plt.plot(column[0],column[1])

    row = GetNthRow(data2, yvalue)
    rowLineplot = plt.subplot(1,2,2)
    plt.title(f'Row-Pixel Saturation at Y={yvalue}')
    plt.xlabel('X Index')
    plt.ylabel('Pixel Value')
    plt.plot(row[0],row[1])

    plt.show()

def gauss_derivative():

    baseFits_path = '/media/al-chromebook/USB20FD/Python/Research/fits/'
    spits_isofield = 'Spitzer24_IsoFields.fits'

    data1 = fits.open(baseFits_path + spits_isofield)[0].data
    data2 = np.diff(data1, 2)

    spits_point_sources = [
        [52,115],
        [35,35],
        [288, 328],
        [258, 337],
        [250, 81],
        [80, 89],
        [90, 309],
        [28, 327]
    ]

    coords = spits_point_sources[0]
    # coords = 227, 154
    cutoutSize = 50
    
    # file_hdu = fits.open(file)[0]
    # file_data = file_hdu.data
    file_data = data2

    from astropy.nddata import Cutout2D 
    import astropy.stats as stats
    from astropy.modeling import models, fitting
    import warnings
    from astropy.utils.exceptions import AstropyUserWarning
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


    plt.figure(figsize=(8, 2.5))
    plt.imshow(Data, origin='lower', interpolation='nearest')
    plt.title("Data")

    plt.show()


def smooth_sofia():
    from astropy.stats import gaussian_fwhm_to_sigma
    from os.path import exists
    from math import floor
    from random import gauss
    import numpy as np

    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans, convolve_fft

    baseFits_path = '/media/al-chromebook/USB20FD/Python/Research/fits/'
    spits_isofield = 'Spitzer24_IsoFields.fits'

    data1 = fits.open(baseFits_path + spits_isofield)[0].data
    data2 = np.diff(data1, 2)

    spits_point_sources = [
        [52,115],
        [35,35],
        [288, 328],
        [258, 337],
        [250, 81],
        [80, 89],
        [90, 309],
        [28, 327]
    ]

    coords = spits_point_sources[0]
    fwhm = 6.9
    # coords = 227, 154
    cutoutSize = 50
    file_data = data2

    from astropy.nddata import Cutout2D 
    import astropy.stats as stats
    from astropy.modeling import models, fitting
    import warnings
    from astropy.utils.exceptions import AstropyUserWarning
    x,y = np.mgrid[:cutoutSize, :cutoutSize]
    Data = Cutout2D(file_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(Data)
    Data -= median

    sigma = fwhm * 1.302 * gaussian_fwhm_to_sigma ## converts from arcseconds to pixels
    kernel = Gaussian2DKernel(sigma)
    astropy_conv = convolve_fft(Data, kernel, allow_huge=True)
    
    # file_hdu = fits.open(file)[0]
    # file_data = file_hdu.data
    plt.figure(figsize=(8, 2.5))
    plt.imshow(Data, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.show()

smooth_sofia()