import pyqtgraph as pg
import numpy as np
from plot_utils import ImpTune
pg.mkQApp()

imp_tuner = ImpTune(dataset="g")

if __name__ == "__main__":
    imp_tuner.show()
    pg.exec()
