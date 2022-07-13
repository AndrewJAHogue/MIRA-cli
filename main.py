import openpyxl as pe
import gaussian_fit
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# __________________Parameters______________________________________________________________
import computer_path
computer_path = computer_path.get_computer_path()
Combined_path = f'{computer_path}/Python/Research/fits/Combined Maps/'
basePath = f'{computer_path}/Python/Research/fits/Full Maps/'


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
    base_figPath = f'{computer_path}/MIRA-CLI/Figures/'

    spits_isofield = 'Spitzer24_IsoFields.fits'

    filename = spits_isofield + '_Gauss2D'
    figPath = f'{computer_path}/MIRA-CLI/Figures/{filename}'

    json_file = figPath +  '/data.json'
    avg =  avg_fwhm(json_file)
    std = std_of_fwhm(json_file)
    print(f'{spits_isofield}_average = {avg} +- {std}')
    return avg

def sofia_iso_avg_fwhm():
    base_figPath = f'{computer_path}/MIRA-CLI/Figures/'

    sofia = 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'

    filename = sofia + '_Gauss2D'
    figPath = base_figPath + filename 

    json_file = figPath +  '/data.json'
    avg =  avg_fwhm(json_file)
    std = std_of_fwhm(json_file)
    print(f'{sofia}_average = {avg} +- {std}')
    return avg

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


def diff_map_sofia():
    from astropy.stats import gaussian_fwhm_to_sigma
    from os.path import exists
    from math import floor
    from random import gauss
    import numpy as np

    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans, convolve_fft

    baseFits_path = '/media/al-linux/USB20FD/Python/Research/fits/'
    # baseFits_path = '/media/al-chromebook/USB20FD/Python/Research/fits/'
    spits_isofield = 'Spitzer24_IsoFields.fits'

    original_data = fits.open(baseFits_path + spits_isofield)[0].data
    diff_data = np.diff(original_data, 2)

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
    file_data = diff_data

    from astropy.nddata import Cutout2D 
    import astropy.stats as stats
    from astropy.modeling import models, fitting
    import warnings
    from astropy.utils.exceptions import AstropyUserWarning
    from lineplots import GetNthColumn, GetNthRow
    x,y = np.mgrid[:cutoutSize, :cutoutSize]
    # original map
    org_cutout = Cutout2D(original_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(org_cutout)
    org_cutout -= median
    # differential map
    diff_cutout = Cutout2D(file_data, (coords), cutoutSize).data
    mean, median, tmp = stats.sigma_clipped_stats(diff_cutout)
    diff_cutout -= median

    # sigma = fwhm * 1.302 * gaussian_fwhm_to_sigma ## converts from arcseconds to pixels
    # kernel = Gaussian2DKernel(2)
    # astropy_conv = convolve_fft(diff_cutout, kernel, allow_huge=True)
    
    # file_hdu = fits.open(file)[0]
    # file_data = file_hdu.data
    plt.subplot(121)
    xy = GetNthColumn(original_data, 0)
    x, y = xy
    print(f'{len(x)=} and {len(y)=} ')
    plt.plot(x, y)
    plt.subplot(122)
    x, y = GetNthColumn(diff_data, 0)
    print(f'{len(x)=} and {len(y)=} ')
    plt.plot(x, y)



    plt.show()

def smoothen_sofia():
    from astropy.stats import gaussian_fwhm_to_sigma
    from os.path import exists
    from math import floor
    from random import gauss
    import numpy as np

    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans, convolve_fft 

    # baseFits_path = f'{computer_path}/USB20FD/Python/Research/fits/'
    baseFits_path = f'{computer_path}/Python/Research/fits/'
    org_path = 'Full Maps/Originals/'
    spits_full = baseFits_path + org_path + 'gcmosaic_24um.fits'
    sofia_full = baseFits_path + org_path + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
    # spits_isofield = 'Spitzer24_IsoFields.fits'

    original_data = fits.open(sofia_full)[0].data
    # diff_data = np.diff(original_data, 2)

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
    sofia_point_sources = [
        [3115,297],
        [2125,3117],
        [1339,2698],
        [958,3374],
        [1145,3417],
        [1174,4061],
        [1160,4201],
        [542,4930],
        [614,4941],
        [565,5025]
    ] 

    # x, y
    # coords = 3115, 298
    # coords = 791, 4718
    # coords =  3080, 372
    coords = 2213, 3647
    from gaussian_fit import get_fwhm_gauss_file, get_fwhm_gauss_data
    # coords = 227, 154
    cutoutSize = 15



    from astropy.nddata import Cutout2D 
    import astropy.stats as stats
    from lineplots import GetNthColumn, GetNthRow
    # original map
    original_data[np.isnan(original_data)] = 0
    original_data[np.isinf(original_data)] = 0
    # org_cutout -= median
    fwhm = get_fwhm_gauss_data((coords), original_data, cutoutSize)['fwhm']
    
    # fwhm *= 3.2 ## ONLY FOR SPITZER FILES

    kernel = Gaussian2DKernel(1.15)
    astropy_conv = convolve_fft(original_data, kernel, allow_huge=True) 
    fwhm_conv = get_fwhm_gauss_data((coords), astropy_conv, cutoutSize)['fwhm']
    # fwhm_conv *= 3.2 ## ONLY FOR SPITZER FILES

    
    org_cutout = Cutout2D(original_data, (coords), cutoutSize).data
    conv_cutout = Cutout2D(astropy_conv, (coords), cutoutSize).data

    # mean, median, tmp = stats.sigma_clipped_stats(org_cutout)
    x, y = GetNthColumn(org_cutout, round(cutoutSize/2))
    plt.subplot(121)
    plt.title(f'Original Data {fwhm=}')
    plt.plot(x, y)

    x, y = GetNthColumn(conv_cutout, round(cutoutSize/2))
    plt.subplot(122)
    plt.title(f'Convolved {fwhm_conv=}')
    plt.plot(x, y)

    print(f'Percent difference = {100 - fwhm/fwhm_conv * 100 =}')


    # plt.subplot(131)
    # plt.imshow(org_cutout)
    # plt.subplot(132)
    # plt.imshow(conv_cutout)
    # plt.subplot(133)
    # plt.imshow(org_cutout - conv_cutout)


    plt.show()

def get_json_info(filepath):
    new_dict = []
    import json

    json_file = computer_path + 'MIRA-CLI/Figures/' + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits_Gauss2D/' + 'data.json'
    with open(filepath, 'r+') as f:
        f_data = json.load(f)

        for key in f_data:
            new_dict.append(key)
    return new_dict

# np.average(org_fwhms)=5.015014581927903 and np.average(conv_fwhms)=6.4360830539570335
def loop_smoothen_sofia():
    from astropy.stats import gaussian_fwhm_to_sigma
    from os.path import exists
    from math import floor
    from random import gauss
    import numpy as np

    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans, convolve_fft 

    # baseFits_path = f'{computer_path}/USB20FD/Python/Research/fits/'
    baseFits_path = f'{computer_path}/Python/Research/fits/'
    org_path = 'Full Maps/Originals/'
    sofia_full = baseFits_path + org_path + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'

    original_data = fits.open(sofia_full)[0].data

    sofia_point_sources = [
        [3115,297],
        [1339,2698],
        [1145,3417],
        [1174,4061],
        [1160,4201],
        [542,4930],
        [614,4941],
        [565,5025]
    ] 
    new_sofia_points = [
        [ 542,  4930 ],
        [ 1123, 4156 ],
        [ 1313, 4132 ],
        [ 1174, 4061 ],
        [ 1105, 3833 ],
        [ 1464, 2917 ],
        [ 2740, 1789 ],
        [ 2614, 1899 ],
        [ 2770, 1864 ],
        [ 2725, 1414 ],
        [ 2693, 1204 ],
        [ 2817, 1214 ]
    ]

    # x, y
    # coords = 3115, 298
    # coords = 791, 4718
    # coords =  3080, 372
    coords = new_sofia_points[0]
    from gaussian_fit import get_fwhm_gauss_file, get_fwhm_gauss_data
    cutoutSize = 15



    from astropy.nddata import Cutout2D 
    import astropy.stats as stats
    from lineplots import GetNthColumn, GetNthRow
    # original map
    # original_data[np.isnan(original_data)] = 0
    # original_data[np.isinf(original_data)] = 0
    kernel = Gaussian2DKernel(1.20)
    astropy_conv = convolve_fft(original_data, kernel, allow_huge=True)
    coadd_data = np.nansum([astropy_conv, original_data], axis=0)  + 0.05
    # org_cutout -= median
    org_fwhms = []
    conv_fwhms = []
    coadd_fwhms = []
    for point in new_sofia_points:
        print(f'\n{point=}')
        fwhm = get_fwhm_gauss_data((point), original_data, cutoutSize)['fwhm']
        org_fwhms.append(fwhm)

        print('\nAnd the convolved point data is::')
        fwhm_conv = get_fwhm_gauss_data((point), astropy_conv, cutoutSize)['fwhm']
        fwhm_coadd = get_fwhm_gauss_data((point),coadd_data , cutoutSize)['fwhm']
        conv_fwhms.append(fwhm_conv)
        coadd_fwhms.append(fwhm_coadd)
        # fwhm_conv *= 3.2 ## ONLY FOR SPITZER FILES

    
    print(f'\n{np.nanmean(org_fwhms)=} and {np.nanmean(conv_fwhms)=} and {np.nanmean(coadd_fwhms)=}')


def coadd():
    from astropy.stats import gaussian_fwhm_to_sigma
    from os.path import exists
    from math import floor
    from random import gauss
    import numpy as np

    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans, convolve_fft 

    # baseFits_path = f'{computer_path}/USB20FD/Python/Research/fits/'
    baseFits_path = f'{computer_path}/Python/Research/fits/'
    org_path = 'Full Maps/Originals/'
    spits_full = baseFits_path + org_path + 'gcmosaic_24um.fits'
    spits_fitted = baseFits_path + 'Full Maps/Spitzer_GCmosaic_24um_onFORCASTheader_JyPix.fits'
    sofia_full = baseFits_path + org_path + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
    # spits_isofield = 'Spitzer24_IsoFields.fits'

    original_data = fits.open(sofia_full)[0].data
    spitz_data = fits.open(spits_fitted)[0].data

    sofia_point_sources = [
        [3115,297],
        [2125,3117],
        [1339,2698],
        [958,3374],
        [1145,3417],
        [1174,4061],
        [1160,4201],
        [542,4930],
        [614,4941],
        [565,5025]
    ] 
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
    # x, y
    # coords = 3115, 298
    # coords = 791, 4718
    # coords =  3080, 372
    coords = sofia_point_sources[0]
    from gaussian_fit import get_fwhm_gauss_file, get_fwhm_gauss_data
    # coords = 227, 154
    cutoutSize = 15



    from astropy.nddata import Cutout2D 
    import astropy.stats as stats
    from lineplots import GetNthColumn, GetNthRow
    # original map
    # original_data[np.isnan(original_data)] = 0
    # original_data[np.isinf(original_data)] = 0
    # org_cutout -= median
    fwhm = get_fwhm_gauss_data((coords), original_data, cutoutSize)['fwhm']
    
    # fwhm *= 3.2 ## ONLY FOR SPITZER FILES

    kernel = Gaussian2DKernel(1.15)
    astropy_conv = convolve_fft(original_data, kernel, allow_huge=True) 
    fwhm_conv = get_fwhm_gauss_data((coords), astropy_conv, cutoutSize)['fwhm']
    coadd_data = np.nansum([astropy_conv, original_data], axis=0)  + 0.05
    # fwhm_conv *= 3.2 ## ONLY FOR SPITZER FILES

    
    # org_cutout = Cutout2D(original_data, (coords), cutoutSize).data
    # conv_cutout = Cutout2D(astropy_conv, (coords), cutoutSize).data

    coords = 490, 4815
    # coords = spits_point_sources[3]
    spits_cutout = Cutout2D(spitz_data, (coords), cutoutSize).data
    coadd_cutout = Cutout2D(coadd_data, (coords), 100).data
    # x, spits_y = GetNthColumn(spitz_data, coords[0])
    # plt.plot(x, spits_y, label='Spitzer')

    # x, conv_y = GetNthColumn(astropy_conv, coords[0])
    # plt.plot(x, conv_y, label='Convolved Sofia')
    # x, coadd_y = GetNthColumn(coadd_data, coords[0])
    # plt.plot(x, coadd_y, label='Coadd')
    # plt.legend()

    get_fwhm_gauss_data((coords),spitz_data, cutoutSize)['fwhm']
    get_fwhm_gauss_data((coords),coadd_data, 100)['fwhm']
    # print(f'Percent difference = {100 - fwhm/fwhm_conv * 100 =}')

    plt.subplot(131)
    plt.imshow(spits_cutout)
    plt.subplot(132)
    plt.imshow(coadd_cutout)


    plt.show()

def saveCoadd():
    from astropy.stats import gaussian_fwhm_to_sigma
    from os.path import exists
    from math import floor
    from random import gauss
    import numpy as np

    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans, convolve_fft 

    # baseFits_path = f'{computer_path}/USB20FD/Python/Research/fits/'
    baseFits_path = f'{computer_path}/Python/Research/fits/'
    org_path = 'Full Maps/Originals/'
    spits_full = baseFits_path + org_path + 'gcmosaic_24um.fits'
    spits_fitted = baseFits_path + 'Full Maps/Spitzer_GCmosaic_24um_onFORCASTheader_JyPix.fits'
    sofia_full = baseFits_path + org_path + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
    # spits_isofield = 'Spitzer24_IsoFields.fits'

    original_data = fits.open(sofia_full)[0].data
    spitz_data = fits.open(spits_fitted)[0].data

    kernel = Gaussian2DKernel(1.15)
    astropy_conv = convolve_fft(original_data, kernel, allow_huge=True) 

    coadd_data = np.nansum([astropy_conv, spitz_data], axis=0)


# np.average(org_fwhms)=5.015014581927903 and np.average(conv_fwhms)=6.4360830539570335


from astropy.io import fits

# baseFits_path = f'{computer_path}/USB20FD/Python/Research/fits/'
baseFits_path = f'{computer_path}/Python/Research/fits/'
org_path = 'Full Maps/Originals/'
sofia_full = baseFits_path + org_path + 'F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'
spitz_path = baseFits_path + org_path + 'Spitzer_GCmosaic_24um_onFORCASTheader_JyPix.fits'
coadd_path = f'{computer_path}/1.15_coadd.fits'

spits_data = fits.open(spitz_path)[0].data
coadd_data = fits.open(coadd_path)[0].data

from lineplots import GetNthColumn, GetNthRow
x, y = GetNthRow(spits_data, 3309)
plt.plot(x, y, label='Spitzer')

x, y = GetNthRow(coadd_data, 3309)
plt.plot(x, y, label='Coadd')
plt.legend()

plt.show()
