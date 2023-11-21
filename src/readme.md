# Assignment 5: Computational Photography



## 1.1 Uncalibrated Photometric Stereo

Run the following command to read the TIF files, perform uncalibrated stereo, calculate and save Albedos, Normals, Depth images and reconstruct a 3D surface in matplotlib. The following file can also be used to tweak GBR and sigma values for enforcing integrability.

```python a1.py```

## 1.2 Calibrated Photometric Stereo

The following command reads the image dataset and the lighting directions and performs calibrated photometric stereo. It saves Albedos, Normals, Depth images and reconstruct a 3D surface in matplotlib

```python a1_calib.py```

## 2. Uncalibrated Photometric Stereo on your own images

Tethering is used to capture images from the camera. Following script is used for the same.

```./script.sh```

The `NEF` images from the camera are converted to `TIFF` format using the following command

```dcraw -r 2.39 1 1.2239 1 -o 1 -q 1 -T -4 ./custom10/*.nef```

The following file can be used to perform uncalibrated stereo on custom images. It takes data from a `custom` folder that has images in `TIFF` format. 

```python a2.py```

