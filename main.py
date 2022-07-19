from settings import *
import numpy as np
import pydicom as pd
import matplotlib
from matplotlib import pyplot
import os
import SimpleITK as sitk
#import radiomics

def open_dicom(dir, start, stop):
    os.chdir(dir)
    dicom_filenames = np.array([f'IMG-0001-000{i}.dcm' for i in range(start, stop)])
    dicom_count = dicom_stop - dicom_start
    dicom_files = np.array([pd.dcmread(dicom_filenames[i]) for i in range(0, dicom_count)])
    return dicom_files

def open_nrrd(dir, name):
    os.chdir(dir)
    image = sitk.ReadImage(name)
    return image

def get_array(nrrd_img):
    pix_array = sitk.GetArrayFromImage(nrrd_img)
    return pix_array

def first_order():
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask)
    firstOrderFeatures.calculateFeatures()
    for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
        print("\t%s: %s" % (key, val))

def plot(data):
    if str(type(data)) == "<class 'pydicom.dataset.FileDataset'>":
        X = data.pixel_array
    else:
        X = data
    pyplot.imshow(X, cmap=matplotlib.cm.gray)
    pyplot.show()


if __name__ == "__main__":
    image_dicom = open_dicom(dicom_dir, dicom_start, dicom_stop)

    mask_nrrd = open_nrrd(mask_dir, mask_name)
    image_nrrd = open_nrrd(nrrd_dir, nrrd__image_name)

    mask_nrrd_array = get_array(mask_nrrd)
    image_nrrd_array = get_array(image_nrrd)

    #plot(image_dicom[10])
    plot(image_nrrd_array[10])

    print(type(image_dicom), type(image_dicom[10]), type(image_nrrd), type(image_nrrd_array))
    #print(len(mask_nrrd_array), len(mask_nrrd_array[0]), len(mask_nrrd_array[0][0]))
    #print(image_nrrd_array[8][200])
    #print(list(dicom_files[0][0x20, 0x32])[2])


#[-247.800, -248.500, -64.000]
#[-247.800, -248.500, -65.000]

#print (dicom_images)
#print(dicom_filenames)

#mask = struct_to_mask(dicom_dir, dicom_files, struct_name)