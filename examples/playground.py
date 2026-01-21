import matplotlib.pyplot as plt

import numpy as np
import cupy

from hillshade import hillshade_gpu
from viewshed import viewshed_gpu

from rtxpy import RTX

import xarray as xr


terrain = xr.open_dataarray("crater_lake_national_park.tif").squeeze()
terrain = terrain[::2, ::2]
terrain.data = terrain.data * 0.2  # scale down
azimuth = 225


def debug(x, y):
    rays = cupy.float32([x, y, 10000, 0, 0, 0, -1, np.inf])
    hits = cupy.float32([0, 0, 0, 0])
    optix = RTX()
    res = optix.trace(rays, hits, 1)
    return res


def onclick(event):
    """
    Click handler for live adjustment of the viewshed origin
    """
    ix, iy = event.xdata, event.ydata
    print('x = {}, y = {}'.format(ix, iy))

    nix = ix/terrain.shape[1]
    niy = iy/terrain.shape[0]

    x_coords = terrain.indexes.get('x').values
    y_coords = terrain.indexes.get('y').values
    rangex = x_coords.max() - x_coords.min()
    rangey = y_coords.max() - y_coords.min()

    global vsw, vsh
    vsw = x_coords.min() + nix*rangex
    vsh = y_coords.max() - niy*rangey

    # debug(ix, iy)
    return None


def test():
    runs = 360
    H, W = terrain.data.shape
    if isinstance(terrain.data, np.ndarray):
        terrain.data = cupy.array(terrain.data)

    fig = plt.figure()
    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    colors = np.uint8(np.zeros((H, W, 3)))
    imgplot = plt.imshow(colors)

    x_coords = terrain.indexes.get('x').values
    y_coords = terrain.indexes.get('y').values
    midx = x_coords.min() + (x_coords.max() - x_coords.min()) / 2
    midy = y_coords.min() + (y_coords.max() - y_coords.min()) / 2

    global vsw, vsh
    global azimuth
    vsw = midx
    vsh = midy

    import time
    for i in range(runs):
        print("Begin Frame ", i)
        azimuth = azimuth + 5
        if (azimuth > 360):
            azimuth -= 360

        beforeRT = time.time()
        hs = hillshade_gpu(terrain,
                           shadows=True,
                           azimuth=azimuth,
                           angle_altitude=25)
        vs = viewshed_gpu(terrain,
                          x=vsw,
                          y=vsh,
                          observer_elev=0.01)
        afterRT = time.time()
        print("  RT took ", afterRT-beforeRT)

        img = np.uint8(hs.data.get()*225)

        withViewshed = True
        if withViewshed:
            visBuf = np.uint8(vs.data.get() > 0) * 255
            view = np.maximum(visBuf, img)
        else:
            view = img
        colors[:, :, 0] = view
        colors[:, :, 1] = img
        colors[:, :, 2] = img
        imgplot.set_data(colors)
        plt.pause(0.001)

    plt.show()

    return


res = test()

print("Done")
