try:
    import Image
except:
    from PIL import Image

import FPI

fname = 'minime90/mrh/2015/20150615/Images/MRH_L_20150615_231355_001.img'

d = FPI.ReadIMG(fname)
d.show()