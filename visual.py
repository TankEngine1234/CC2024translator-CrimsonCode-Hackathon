from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, OEM
import csv

api = PyTessBaseAPI(path="tessdata-4.1.0",psm=PSM.AUTO_OSD,oem=OEM.DEFAULT)

def init():
    image = Image.open("current.png")
    api.SetImage(image)
    api.Recognize()
    
def get_orientation():
    it = api.AnalyseLayout()
    orientation, direction, order, deskew_angle = it.Orientation()
    return format(orientation), format(deskew_angle)

def write_TSV():
    with open("results.tsv", "w") as result:
        result.write(api.GetTSVText(0))
        result.close()

def get_regions():
    regions = []
    with open("results.tsv", "r") as tsv:
        f = csv.reader(tsv, delimiter='\t', quotechar='"')
        i = 0
        str = ''
        left = []
        top = []
        right = []
        bottom = []
        for row in f:
            if float(row[2]) == i:
                if float(row[10]) == 0 or float(row[10]) > 85:
                    str = " ".join((str, row[11]))
                    left.append(float(row[6]))
                    top.append(float(row[7]))
                    right.append(float(row[6]) + float(row[8]))
                    bottom.append(float(row[7]) + float(row[9]))
            else:
                regions.append((str, min(left, default=-1), min(top, default=-1), max(right, default=-1), max(bottom, default=-1)))
                str = ''
                left = []
                top = []
                right = []
                bottom = []
                i = i + 1
    return regions

                
