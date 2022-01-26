import numpy as np
import matplotlib.pyplot as plt

"""
Represent an image using a 2D Fourier series

full color image animation:
ffmpeg -framerate 10 -i fourier_%d.png -c:v libx264 -vf "fps=60" -crf 10 -c:a aac fourier.mp4

rgb images animation:
ffmpeg -framerate 10 -i fourier_combined_%d.png -c:v libx264 -vf "fps=60" -crf 10 -c:a aac fourier_combined.mp4

fourier mode animation:
ffmpeg -framerate 2 -start_number 0 -i anim_%d.png -t 0:12.5 -c:v libx264 -vf "fps=60" -crf 10 -c:a aac anim1.mp4
ffmpeg -framerate 14 -start_number 25 -i anim_%d.png -t 0:4.0 -c:v libx264 -vf "fps=60" -crf 10 -c:a aac anim2.mp4
ffmpeg -framerate 44 -start_number 81 -i anim_%d.png -t 0:2.0 -c:v libx264 -vf "fps=60" -crf 10 -c:a aac anim3.mp4
ffmpeg -framerate 60 -start_number 169 -i anim_%d.png -c:v libx264 -vf "fps=60" -crf 10 -c:a aac anim4.mp4
"""


do_save = False
do_anim = False


def plot_coeffs(mode, coeffs, img, anim_count, fig_no=3, name=None, compression=None, N=None, k=0, l=0, center=None):
    # 4 plots side-by-side
    # plot 1: fourier mode in grayscale
    # plot 2: absolute value of fourier coefficients (transformed by sqrt function)
    # plot 3: complex angle of the fourier coeffients
    # plot 4: image reconstruction
    
    fig = plt.figure(fig_no,
                     dpi=240,
                     figsize=(16, 9))
    plt.clf()

    rows = 1
    columns = 4

    imw = max(N+1, 3)

    # plot mode
    fig.add_subplot(rows, columns, 1)
    plt.imshow(mode)
    plt.axis('off')

    fig.add_subplot(rows, columns, 2)

    # plot abs
    transf = np.sqrt(np.abs(coeffs))
    plt.imshow(1-transf[center-imw:center+imw, center-imw:center+imw])
    plt.axis('off')

    # plot ang
    fig.add_subplot(rows, columns, 3)
    plt.imshow(1 - (np.angle(coeffs[center-imw:center+imw, center-imw:center+imw]) % (2*np.pi)) / (2*np.pi))
    plt.axis('off')
    
    fig.add_subplot(rows, columns, 4)
    plt.imshow(img)
    plt.axis('off')

    if N is not None:
        # breakpoint()
        plt.gcf().text(0.5, 0.0, "N=%d\n mode=(%d, %d)\n compression=%.4f%%" % (N, k, l, 100*compression),
                       fontsize=36,
                       horizontalalignment='center',
                       verticalalignment='bottom')

    name = "anim/anim_%d.png" % (anim_count)
    plt.savefig(name)


def plot_all(img, fig_no=1, name=None, label=None):
    # two plots
    # plot 1: (red, green, blue, full) image reconstructions
    # plot 2: full image reconstruction
    
    img_r = img+0
    img_g = img+0
    img_b = img+0

    img_r[:, :, 1:] = 0
    img_g[:, :, 0] = 0
    img_g[:, :, 2] = 0
    img_b[:, :, 0:2] = 0

    rows = 1
    columns = 4

    # combined
    fig = plt.figure(fig_no,
                     dpi=240,
                     figsize=(16, 9))
    plt.clf()

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_r)
    plt.axis('off')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(img_g)
    plt.axis('off')

    fig.add_subplot(rows, columns, 3)
    plt.imshow(img_b)
    plt.axis('off')

    fig.add_subplot(rows, columns, 4)
    plt.imshow(img)
    plt.axis('off')

    if label is not None:
        plt.gcf().text(0.5, 0.0, "N=%d" % (label),
                       fontsize=36,
                       horizontalalignment='center',
                       verticalalignment='bottom')

    if name is not None and do_save:
        plt.savefig(name.replace("*", "_combined")+".png")

    # single
    fig = plt.figure(fig_no,
                     dpi=240,
                     figsize=(16, 9))
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    if label is not None:
        plt.gcf().text(0.5, 0.0, "N=%d" % (label),
                       fontsize=36,
                       horizontalalignment='center',
                       verticalalignment='bottom')

    if name is not None and do_save:
        plt.savefig(name.replace("*", "")+".png")


def square_boundary(N):
    # give integer pairs representing the
    # coordinates of boundary of the square defined by
    # vertices (N, N), (-N, N), (N, -N), (-N, -N)

    i = N
    j = N

    pairs = []

    j = N
    for i in range(N, -N-1, -1):
        pairs.append((i, j))

    i = -N
    for j in range(N-1, -N-1, -1):
        pairs.append((i, j))

    j = -N
    for i in range(-N+1, N+1, 1):
        pairs.append((i, j))

    i = N
    for j in range(-N+1, N, 1):
        pairs.append((i, j))

    return pairs
    

def main():
    # number of modes to use
    N = 100

    # original image
    img_dir = 'pikachu.jpg'

    # plot image
    img = plt.imread(img_dir)
    plot_all(img, fig_no=1, name="figs/og*")

    # get image shape and size
    npx, npy, d = np.shape(img)
    total_pixels = npx*npy

    # create a mesh for the pixels
    xv = np.linspace(0, npx, npx+1) / npx
    yv = np.linspace(0, npy, npy+1) / npy

    # reconstruction of image
    val = np.zeros((npx, npy, 3))
    # fourier mode
    mode = np.zeros((npx, npy, 3))
    # fourier coefficients
    coeffs = np.zeros((2*N+1, 2*N+1, 3), dtype=complex)
    center = N

    # half points of pixel squares
    xvh = 0.5 * (xv[+1:]+xv[:-1])
    yvh = 0.5 * (yv[+1:]+yv[:-1])

    # grid spacing
    dx = 1 / npx
    dy = 1 / npy

    count = 0
    anim_count = 0

    # loop through all combos of k and l
    av = [[], [], []]
    for n in range(N+1):
        # get integer pairs on the square defined by n
        pairs = square_boundary(n)
        # loop through each pair and compute fourier coefficient
        for l, k in pairs:
            print((k, l), flush=True)

            # term: is the fourier coefficient
            # the (k, l)'th coefficient is given by
            # 1 / (Lx Ly) \int_{0}^{Lx} \int_{0}^{Ly} c(x, y) exp(-2 pi i k x / Lx) exp(-2 pi i l y / Ly) dx dy
            # where c(x, y) is the image.
            # This is computed exactly using the fact that images are constant over a pixel domain.
            # (This could be probably be sped up significantly using the DFT.)

            hv = (+ np.exp(- 2 * np.pi * 1j * l * yv[+1:])
                  - np.exp(- 2 * np.pi * 1j * l * yv[:-1]))
            gv = (+ np.exp(- 2 * np.pi * 1j * k * xv[+1:])
                  - np.exp(- 2 * np.pi * 1j * k * xv[:-1]))

            qv = np.exp(2 * np.pi * 1j * l * yvh)
            pv = np.exp(2 * np.pi * 1j * k * xvh)
            pv_qv = np.outer(pv, qv)

            # compute for r, g, and b
            for dim in range(3):
                if k == 0:
                    if l == 0:
                        term = dx * dy * np.sum( img[:, :, dim] )
                    else:
                        term = - dx / (2 * np.pi * 1j * l) \
                            * np.sum( img[:, :, dim] *  hv )
                else:
                    if l == 0:
                        term = - dy / (2 * np.pi * 1j * k) \
                            * np.sum( img[:, :, dim].T * gv )
                    else:
                        term = - 1 / (4 * np.pi ** 2 * k * l) \
                            * np.sum( (img[:, :, dim] * hv).T * gv )

                av[dim].append(term)

                coeffs[center+k, center+l, dim] = term

                val[:, :, dim] += \
                    np.real(
                        + (av[dim][count] \
                           * pv_qv)) / 255

            count += 1

            # plot
            name = "figs/coeffs_%d" % (count)
            compression = 2*count / total_pixels
            if do_anim:

                new_val = np.clip(val, 0, 1)

                if k == 0 and l == 0:
                    start = (np.real(pv_qv)+1)/2*0
                else:
                    start = (np.real(pv_qv)+1)/2
                for dim in range(3):
                    mode[:, :, dim] = start

                plot_val = new_val
                plot_coeffs(mode, coeffs / 255, plot_val, anim_count,
                            fig_no=3, compression=compression, N=n, k=k, l=l, center=center)
                anim_count += 1

        # plot
        name = "figs/fourier*_%d" % (n)
        plot_all(np.clip(val, 0, 1), fig_no=2, name=name, label=n)


if __name__ == "__main__":
    main()
