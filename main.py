import numpy as np
import matplotlib.pyplot as plt


def plot(img, figs=[1,2,3,4], name=None):
    
    img_r = img+0
    img_g = img+0
    img_b = img+0

    img_r[:, :, 1:] = 0
    img_g[:, :, 0] = 0
    img_g[:, :, 2] = 0
    img_b[:, :, 0:2] = 0

    plt.figure(figs[0])
    plt.clf()
    plt.imshow(img)
    if name is not None:
        plt.savefig(name.replace("*", "")+".png")

    plt.figure(figs[1])
    plt.clf()
    plt.imshow(img_r)
    if name is not None:
        plt.savefig(name.replace("*", "r")+".png")

    plt.figure(figs[2])
    plt.clf()
    plt.imshow(img_g)
    if name is not None:
        plt.savefig(name.replace("*", "g")+".png")

    plt.figure(figs[3])
    plt.clf()
    plt.imshow(img_b)
    if name is not None:
        plt.savefig(name.replace("*", "b")+".png")

def square_boundary(N):

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
    img_dir = 'pikachu.jpg'
    img = plt.imread(img_dir)
    plot(img)

    npx, npy, d = np.shape(img)
    
    xv = np.linspace(0, npx, npx+1) / npx
    yv = np.linspace(0, npy, npy+1) / npy

    dx = 1 / npx
    dy = 1 / npy

    dim = 2

    N = 10

    # loop through all combos of k and l
    av = [[], [], []]
    for n in range(N+1):
        pairs = square_boundary(n)
        for l, k in pairs:
            print((k, l), flush=True)
            
            hv = (+ np.exp(- 2 * np.pi * 1j * l * yv[+1:])
                  - np.exp(- 2 * np.pi * 1j * l * yv[:-1]))
            gv = (+ np.exp(- 2 * np.pi * 1j * k * xv[+1:])
                  - np.exp(- 2 * np.pi * 1j * k * xv[:-1]))

            for dim in range(3):
                if k == 0:
                    if l == 0:
                        term = dx * dy * np.sum( img[:, :, dim] )
                    else:
                        term = - dx / (2 * np.pi * 1j * l) \
                            * np.sum( img[:, :, dim] \
                                      *  hv )
                else:
                    if l == 0:
                        term = - dy / (2 * np.pi * 1j * k) \
                            * np.sum( img[:, :, dim].T \
                                      * gv )
                    else:
                        term = - 1 / (4 * np.pi ** 2 * k * l) \
                            * np.sum( (img[:, :, dim] \
                                       * hv).T
                                      * gv )

                av[dim].append(term)

    # plt.figure()
    # plt.plot(np.abs(np.array(av)))
    
    count = 0
    xvh = 0.5 * (xv[+1:]+xv[:-1])
    yvh = 0.5 * (yv[+1:]+yv[:-1])

    val = np.zeros((npx, npy, 3))
    for n in range(N+1):
        pairs = square_boundary(n)
        for l, k in pairs:
            hv = np.exp(2 * np.pi * 1j * l * yvh)
            gv = np.exp(2 * np.pi * 1j * k * xvh)
            hv_gv = np.outer(gv, hv)
            for dim in range(3):
                val[:, :, dim] += \
                    np.real(
                        + (av[dim][count] \
                           * hv_gv)) / 255
            count += 1

        # plot
        name = ("figs/fourier*_%3d" % (n)).replace(" ", "0")
        plot(val, [5,6,7,8], name)
        

    # plt.show()

    


if __name__ == "__main__":
    main()
