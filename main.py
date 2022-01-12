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

    N = 3

    # loop through all combos of k and l
    av = []
    for labs in range(N+1):
        lv = [-labs, labs]
        if labs == 0:
            lv = [labs]
        for l in lv:
            hv = (+ np.exp(- 2 * np.pi * 1j * l * yv[+1:])
                  - np.exp(- 2 * np.pi * 1j * l * yv[:-1]))
            for kabs in range(N+1):
                kv = [-kabs, kabs]
                if kabs == 0:
                    kv = [kabs]
                for k in kv:
                    print((k, l), flush=True)

                    if k == 0:
                        if l == 0:
                            term = dx * dy * np.sum( img[:, :, dim] )
                        else:
                            term = - dx / (2 * np.pi * 1j * l) \
                                * np.sum( img[:, :, dim] \
                                          *  hv )
                    else:
                        gv = (+ np.exp(- 2 * np.pi * 1j * k * xv[+1:])
                              - np.exp(- 2 * np.pi * 1j * k * xv[:-1]))
                        if l == 0:
                            term = - dy / (2 * np.pi * 1j * k) \
                                * np.sum( img[:, :, dim].T \
                                          * gv )
                        else:
                            term = - 1 / (4 * np.pi ** 2 * k * l) \
                                * np.sum( (img[:, :, dim] \
                                           * hv).T
                                          * gv )

                    av.append(term)

    plt.figure()
    plt.plot(np.abs(np.array(av)))
    
    count = 0
    xvh = 0.5 * (xv[+1:]+xv[:-1])
    yvh = 0.5 * (yv[+1:]+yv[:-1])

    val = np.zeros((npx, npy, 3))
    for labs in range(N+1):
        lv = [-labs, labs]
        if labs == 0:
            lv = [labs]
        for l in lv:

            hv = np.exp(2 * np.pi * 1j * l * yvh)
            for kabs in range(N+1):
                kv = [-kabs, kabs]
                if kabs == 0:
                    kv = [kabs]
                for k in kv:

                    gv = np.exp(2*np.pi*1j*k*xvh)
                    val[:, :, dim] += \
                        np.real(
                            + (av[count] \
                               * np.outer(gv, hv))) / 255
                    count += 1

        # plot
        name = ("figs/fourier*_%3d" % (labs)).replace(" ", "0")
        plot(val, [5,6,7,8], name)
        

    # plt.show()

    


if __name__ == "__main__":
    main()
