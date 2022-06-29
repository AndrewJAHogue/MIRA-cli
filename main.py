import openpyxl as pe
import gaussian_fit


def csv():
    basePath = '/media/al-chromebook/USB20FD/'
    # basePath = '/media/al-linux/USB20FD/'
    filename = "SOFIA pointsources"
    data = pe.load_workbook(f'{basePath}{filename}.xlsx')
    sheet = data.active
    # x_cells, y_cells = sheet['B'][49:52], sheet['C'][49:52]
    x_cells, y_cells = sheet['B'][49:101], sheet['C'][49:101]

    for index_x, x in enumerate(x_cells):
        if 'Corrected' in sheet['I'][index_x + 49].value:
            coords = (x.value, y_cells[index_x].value)
            print(f'{coords=}')
            gaussian_fit.MoffatFit(coords)

        
        
csv()