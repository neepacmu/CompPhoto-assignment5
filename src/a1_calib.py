import os
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from cp_hw5 import integrate_poisson, integrate_frankot, load_sources


def read_data(data_dir):

    output = []

    #for img in os.listdir(data_dir):
    for i in range(1,8):

        img = f'input_{str(i)}.tif'
        print(img)
        if '.tif' not in img:
            continue

        file_path = os.path.join(data_dir, img)
        raw_img = skimage.io.imread(file_path)/65535.0
        
        raw_img = raw_img*255.0

        raw_img = raw_img.astype('float32')
        print(raw_img.max(), raw_img.min())

        xyz_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2XYZ)

        l_xyz = xyz_img[:,:,1]

        output.append(l_xyz)
    
    return np.array(output)


def integrable_normals_delta(be, sigma = 7):

    be = be.reshape((3, height, width)).transpose(1, 2, 0)

    blurred_be = np.zeros_like(be)

    blurred_be[:,:,0] = gaussian_filter(be[:,:,0], sigma=sigma)
    blurred_be[:,:,1] = gaussian_filter(be[:,:,1], sigma=sigma)
    blurred_be[:,:,2] = gaussian_filter(be[:,:,2], sigma=sigma)

    be_x, be_y = np.gradient(blurred_be, axis = (0,1))

    print(be_x.shape, be_y.shape)

    A1 = be[:, :, 0] * be_x[:, :, 1] - be[:, :, 1] * be_x[:, :, 0]
    A2 = be[:, :, 0] * be_x[:, :, 2] - be[:, :, 2] * be_x[:, :, 0]
    A3 = be[:, :, 1] * be_x[:, :, 2] - be[:, :, 2] * be_x[:, :, 1]
    A4 = -be[:, :, 0] * be_y[:, :, 1] + be[:, :, 1] * be_y[:, :, 0]
    A5 = -be[:, :, 0] * be_y[:, :, 2] + be[:, :, 2] * be_y[:, :, 0]
    A6 = -be[:, :, 1] * be_y[:, :, 2] + be[:, :, 2] * be_y[:, :, 1]

    A = np.stack((A1.flatten(), A2.flatten(), A3.flatten(), A4.flatten(), A5.flatten(), A6.flatten()), axis=1)
    _, S, Vt = np.linalg.svd(A, full_matrices = False)
    x = Vt[-1]  
    print(x)
    #x /= x[-1]  

    #print(np.diag(S)) 
    delta = np.array([[-x[2], x[5], 1],
                    [x[1], -x[4], 0],
                    [-x[0], x[3], 0]])

    
    return delta



if __name__ == "__main__":

    data = read_data('data')

    print(data.shape)
    #data 
    height, width = data.shape[1:3]

    I = data.reshape((7,-1))
    L = load_sources()

    print(I.shape)
    B = np.linalg.lstsq(L, I)

    Be = B[0]
    

    Ae = np.linalg.norm(Be, axis=0)
    Ne = Be / Ae

    Ae_image    = Ae.reshape((height, width))
    Ne_image = Ne.reshape((3, height, width)).transpose(1, 2, 0)
    Be_image = Be.reshape((3, height, width)).transpose(1, 2, 0)

    print(Ae_image.max(), Ae_image.min(), "AE")

    plt.subplot(1, 3, 1)
    plt.imshow(Ae_image, cmap = 'gray')
    plt.title('Albedo (A)')

    cv2.imwrite('out/calib_a.jpg', (Ae_image/Ae_image.max())*255.0)

    print(Ne_image.shape)
    plt.subplot(1, 3, 2)
    Ne_image_dis = (Ne_image + 1)/2

    NQ_image_display = cv2.cvtColor(Ne_image_dis.astype('float32'), cv2.COLOR_RGB2BGR)
    print(NQ_image_display.max(), NQ_image_display.min())
    cv2.imwrite('out/calib_n.jpg', NQ_image_display*255.0)
    plt.imshow(NQ_image_display)
    plt.title('Normals (N)')
    eps = 1e-6


    Be_image[:,:,0] = -Be_image[:,:,0]/(Be_image[:,:,-1] + eps)
    Be_image[:,:,1] = -Be_image[:,:,1]/(Be_image[:,:,-1] + eps)

    output_z = integrate_poisson( Be_image[:,:,0],  Be_image[:,:,1])
    output_z_img = output_z/output_z.max()
    plt.subplot(1, 3, 3)
    plt.imsave('out/calib_d.jpg', output_z_img, cmap = 'gray')
    plt.title('Depth')

    plt.savefig('out/q_calib.jpg')



    x = np.arange(0, output_z.shape[1], 1)
    y = np.arange(0, output_z.shape[0], 1)
    
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    from matplotlib.colors import LightSource
    ls = LightSource()
    color_shade = ls.shade(-output_z, plt.cm.gray)

    surf = ax.plot_surface(X, Y, -output_z, facecolors=color_shade, rstride=4, cstride=4)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
