import os
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from cp_hw5 import integrate_poisson, integrate_frankot


def read_data(data_dir):

    output = []
    n = 1

    for i in range(1,10):

        img = f'img{str(i)}.tiff'
        print(img)
        if '.tif' not in img:
            continue

        file_path = os.path.join(data_dir, img)
        raw_img = skimage.io.imread(file_path)/65535.0
        
        raw_img = raw_img*255.0

        
        raw_img = raw_img[::n, ::n]

        raw_img = raw_img.astype('float32')
        print(raw_img.max(), raw_img.min())

        xyz_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2XYZ)

        l_xyz = xyz_img[:,:,1]

        output.append(l_xyz)
    
    return np.array(output)


def integrable_normals_delta(be, sigma = 5):

    be = be.reshape((3, height, width)).transpose(1, 2, 0)

    blurred_be = np.zeros_like(be)

    blurred_be[:,:,0] = gaussian_filter(be[:,:,0], sigma=sigma)
    blurred_be[:,:,1] = gaussian_filter(be[:,:,1], sigma=sigma)
    blurred_be[:,:,2] = gaussian_filter(be[:,:,2], sigma=sigma)

    be_y, be_x = np.gradient(blurred_be, axis = (0,1))

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

    delta = np.array([[-x[2], x[5], 1],
                    [x[1], -x[4], 0],
                    [-x[0], x[3], 0]])

    return delta


if __name__ == "__main__":

    data = read_data('custom7')
    #data = read_data('custom4')

    height, width = data.shape[1:3]
    n = data.shape[0]

    I = data.reshape((n,-1))
    print(I.shape)
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    print(U.shape, S.shape, Vt.shape)
    Le = U[:, :3] @ np.diag(np.sqrt(S[:3]))

    Be = np.diag(np.sqrt(S[:3])) @ Vt[:3, :]

    Ae = np.linalg.norm(Be, axis=0)
    Ne = Be / Ae
    

    print(Le.shape, Be.shape, Ne.shape, Ae.shape)
    Ae_image = Ae.reshape((height, width))
    Ne_image = Ne.reshape((3, height, width)).transpose(1, 2, 0)


    plt.subplot(1, 2, 2)    

    Q = integrable_normals_delta(Be)

    print(Q)

    LQ = Q @ Le.T
    BQ = Q.T @ Be

    AQ = np.linalg.norm(BQ, axis=0)
    NQ = BQ / AQ

    Ae_image = AQ.reshape((height, width))
    Ne_image = NQ.reshape((3, height, width)).transpose(1, 2, 0)
    Be_image = BQ.reshape((3, height, width)).transpose(1, 2, 0)

    plt.subplot(1, 3, 1)
    plt.imshow(Ae_image*10, cmap = 'gray')
    
    plt.title('Albedo (AQ)')

    plt.subplot(1, 3, 2)
    cv2.imwrite('out/q2_1_a.jpg', (Ae_image/Ae_image.max())*255.0)
    
    Ne_image_dis = (Ne_image + 1)/2

    NQ_image_display = cv2.cvtColor(Ne_image_dis.astype('float32'), cv2.COLOR_RGB2BGR)

    cv2.imwrite('out/q2_1_n.jpg', (Ne_image_dis)*255.0)

    plt.imshow(Ne_image_dis)
    plt.title('Normals (NQ)')
    
    eps = 1e-7

    Be_image[:,:,0] = -Be_image[:,:,0]/(Be_image[:,:,-1] + eps)
    Be_image[:,:,1] = -Be_image[:,:,1]/(Be_image[:,:,-1] + eps)

    output_z = integrate_poisson(Be_image[:,:,0],  Be_image[:,:,1])

    plt.subplot(1, 3, 3)

    output_z_img = output_z/output_z.max()
    plt.imshow(output_z_img, cmap='gray')
    plt.title('Depth')

    plt.imsave('out/q2_1_d.jpg', output_z_img, cmap = 'gray')

    x = np.arange(0, output_z.shape[1], 1)
    y = np.arange(0, output_z.shape[0], 1)
    
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    from matplotlib.colors import LightSource
    ls = LightSource()
    n = 3
    color_shade = ls.shade(-output_z[::n,::n], plt.cm.gray)

    surf = ax.plot_surface(X[::n, ::n], Y[::n, ::n], -output_z[::n, ::n], facecolors=color_shade, rstride=4, cstride=4)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    I_new = LQ[:,2] @ BQ

    temp = I_new.reshape((1, height, width)).transpose(1, 2, 0)
    temp = temp/255.0
    temp = (temp + 1)/2
    plt.imshow(temp, cmap='gray')

    print(I_new.shape, "Hhhhhhhhhhhhh")

    plt.imsave('out/new_view_1.jpg', temp[:,:,0], cmap = 'gray')
    
    plt.show()
