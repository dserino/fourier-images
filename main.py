import numpy as np
import matplotlib.pyplot as plt




def main():
    img_dir = 'pikachu.jpg'
    img = plt.imread(img_dir)

    npx, npy, d = np.shape(img)
    
    img_r = img+0
    img_g = img+0
    img_b = img+0

    img_r[:, :, 1:] = 0
    img_g[:, :, 0] = 0
    img_g[:, :, 2] = 0
    img_b[:, :, 0:2] = 0

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(img_r)
    plt.figure()
    plt.imshow(img_g)
    plt.figure()
    plt.imshow(img_b)
    # plt.show()

    xv = np.linspace(0, npx, npx+1) / npx
    yv = np.linspace(0, npy, npy+1) / npy

    dx = 1 / npx
    dy = 1 / npy

    k = 2
    l = 3

    dim = 2

    Nx = 10
    Ny = 10

    # loop through all combos of k and l
    av = []
    for l in range(-Ny, Ny+1):
        hv = (+ np.exp(- 2 * np.pi * 1j * l * yv[+1:])
              - np.exp(- 2 * np.pi * 1j * l * yv[:-1]))
        for k in range(-Nx, Nx+1):
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

    # breakpoint()
    plt.figure()
    plt.plot(np.abs(np.array(av)))
    
    # breakpoint()
    count = 0
    xvh = 0.5 * (xv[+1:]+xv[:-1])
    yvh = 0.5 * (yv[+1:]+yv[:-1])

    val = np.zeros((npx, npy, 3))
    for l in range(-Ny, Ny+1):
        hv = np.exp(2 * np.pi * 1j * l * yvh)
        for k in range(-Nx, Nx+1):
            gv = np.exp(2*np.pi*1j*k*xvh)
            val[:, :, dim] += \
                np.real(
                    + (av[count] \
                       * np.outer(gv, hv))) / 255
            count += 1
    # breakpoint()
    plt.figure()
    plt.imshow(val)
    plt.show()
    # breakpoint()
    


if __name__ == "__main__":
    main()
