import sys
from settings import *
import numpy as np
import pydicom as pd
import matplotlib
from matplotlib import pyplot
import SimpleITK as sitk
from radiomics import firstorder, shape


def open_dicom(dir: str, start: int, stop: int):
    dicom_filenames = np.array([f'{dir}IMG-0001-000{i}.dcm' for i in range(start, stop)])
    dicom_count = dicom_stop - dicom_start
    dicom_files = np.array([pd.dcmread(dicom_filenames[i]) for i in range(0, dicom_count)])
    return dicom_files


def open_nrrd(dir: str, name: str):
    image = sitk.ReadImage(dir + name)
    return image


def get_array(nrrd_img: sitk.Image):
    pix_array = sitk.GetArrayFromImage(nrrd_img)
    return pix_array


def first_order(image: sitk.Image, mask: sitk.Image):
    first_order_features = firstorder.RadiomicsFirstOrder(image, mask)
    first_order_features.enableAllFeatures()
    first_order_features.execute()
    return first_order_features


def img_to_shape(image: sitk.Image, mask: sitk.Image):
    img_shape = shape.RadiomicsShape(image, mask)
    return img_shape


def plot_array(data: np.ndarray):
    pyplot.imshow(data, cmap=matplotlib.cm.gray)
    pyplot.show()


def get_mean(img_ord: firstorder.RadiomicsFirstOrder):
    count = float(img_ord.getMeanFeatureValue())
    return count


def get_sd(img_ord: firstorder.RadiomicsFirstOrder):
    count = float(img_ord.getStandardDeviationFeatureValue())
    return count


def get_median(img_ord: firstorder.RadiomicsFirstOrder):
    count = float(img_ord.getMedianFeatureValue())
    return count


def get_diam_x(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getMaximum2DDiameterColumnFeatureValue())
    return count


def get_diam_y(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getMaximum2DDiameterRowFeatureValue())
    return count


def get_diam_z(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getMaximum2DDiameterSliceFeatureValue())
    return count


def get_major(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getMajorAxisLengthFeatureValue())
    return count


def get_minor(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getMinorAxisLengthFeatureValue())
    return count


def get_mesh_volume(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getMeshVolumeFeatureValue())
    return count


def get_voxel_volume(img_sh: shape.RadiomicsShape):
    count = float(img_sh.getVoxelVolumeFeatureValue())
    return count


if __name__ == "__main__":
    # image_dicom = open_dicom(dicom_dir, dicom_start, dicom_stop)

    mask_nrrd = open_nrrd(mask_dir, mask_name)
    image_nrrd = open_nrrd(nrrd_dir, nrrd__image_name)

    mask_nrrd_array = get_array(mask_nrrd)
    image_nrrd_array = get_array(image_nrrd)

    img_orders = first_order(image_nrrd, mask_nrrd)
    img_shape = img_to_shape(image_nrrd, mask_nrrd)

    mean = get_mean(img_orders)
    sd = get_sd(img_orders)
    median = get_median(img_orders)
    x = get_diam_x(img_shape)
    y = get_diam_y(img_shape)
    z = get_diam_z(img_shape)
    major = get_major(img_shape)
    minor = get_minor(img_shape)
    mesh_volume = get_mesh_volume(img_shape)
    voxel_volume = get_voxel_volume(img_shape)

    print(f' mean = {mean}\n sd = {sd}\n median = {median}\n x = {x}\n y = {y}\n z = {z}\n major = {major}\n '
          f'minor = {minor}\n mesh volume = {mesh_volume}\n voxel volume = {voxel_volume}')

    plot_array(image_nrrd_array[10])
