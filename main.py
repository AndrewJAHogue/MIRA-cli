import openpyxl as pe
import gaussian_fit


def csv():
    # basePath = '/media/al-chromebook/USB20FD/'
    basePath = '/media/al-linux/USB20FD/'
    filename = "SOFIA pointsources"
    data = pe.load_workbook(f'{basePath}{filename}.xlsx')
    sheet = data.active
    starting_row, starting_column = 49, 2
    ending_row, ending_column = 101, 9

    row, col = starting_row, starting_column
    while row <= ending_row:
        col = starting_column
        while col <= ending_column:
            # print(sheet.cell(row, col).value)
            if 'Corrected' in sheet.cell(row, ending_column).value and type(sheet.cell(row, ending_column).value) != None:
                x = sheet.cell(row, col).value
                col += 1
                y = sheet.cell(row, col).value

                col += 1
                fwhm_major = sheet.cell(row, col).value
                col += 1
                fwhm_minor = sheet.cell(row, col).value
                col += 1
                gPhot = sheet.cell(row, col).value
                col += 1
                phot = sheet.cell(row, col).value
                col += 2
                filename_excel = sheet.cell(row, col).value
                print(f'{x=} and {y=} and {filename_excel=}')
                gaussian_fit.MoffatFit(( x, y ))


                    # col += 1
            break
        row += 1

        
        
csv()