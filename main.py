from settings import *
import numpy as np
import pydicom as pd
import matplotlib
from matplotlib import pyplot
import SimpleITK as sitk
from radiomics import firstorder, featureextractor, shape
import six


def open_dicom(dir: str, start: int, stop: int):
    dicom_filenames = np.array([f'{dir}IMG-0001-000{i}.dcm' for i in range(start, stop)])
    dicom_count = dicom_stop - dicom_start
    dicom_files = np.array([pd.dcmread(dicom_filenames[i]) for i in range(0, dicom_count)])
    return dicom_files


def open_nrrd(dir: str, name: str):
    image = sitk.ReadImage(dir + name)
    return image


def get_array(nrrd_img: sitk.SimpleITK.Image):
    pix_array = sitk.GetArrayFromImage(nrrd_img)
    return pix_array


def first_order(image, mask):
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask)
    firstOrderFeatures.enableAllFeatures()
    firstOrderFeatures.execute()
    return firstOrderFeatures


def img_to_shape(image, mask):
    img_shape = shape.RadiomicsShape(image, mask)
    return img_shape


def img_extractor(img_path: str, mask_path: str):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    result = extractor.execute(img_path, mask_path)
    for (key, val) in six.iteritems(result):
        print(f'{key} : {val}')


def plot(data):
    if str(type(data)) == "<class 'pydicom.dataset.FileDataset'>":
        X = data.pixel_array
    else:
        X = data
    pyplot.imshow(X, cmap=matplotlib.cm.gray)
    pyplot.show()


def get_masked_image(img_array: np.ndarray, mask_array: np.ndarray):
    return np.multiply(img_array, mask_array)


def get_coordinates(mask: np.ndarray):
    coord_array = []
    for slice in range(len(mask)):
        for coloumn in range(len(mask[0])):
            for a in range(len(mask[0][0])):
                if mask[slice][coloumn][a] != 0:
                    coord_array.append([slice, coloumn, a])
    return np.array(coord_array)


def get_mean(img_array: np.ndarray, coord: np.ndarray):
    summary: int = 0
    for i in range(len(coord)):
        slice = coord[i][0]
        coloumn = coord[i][1]
        row = coord[i][2]
        summary += img_array[slice][coloumn][row]
    return summary / len(coord)


def get_sd(img_array: np.ndarray, coord: np.ndarray, mean: float):
    summary: int = 0
    for i in range(len(coord)):
        slice = coord[i][0]
        coloumn = coord[i][1]
        row = coord[i][2]
        summary += np.square(img_array[slice][coloumn][row] - mean)
    return np.sqrt(summary / len(coord))


def get_median(img_array: np.ndarray, coord: np.ndarray):
    summary: list = []
    for i in range(len(coord)):
        slice = coord[i][0]
        coloumn = coord[i][1]
        row = coord[i][2]
        summary.append(img_array[slice][coloumn][row])
    ar = np.array(summary)
    ar = np.sort(ar)
    return np.median(ar)


def get_diams(mask: np.ndarray, ps: np.ndarray):
    min_z = len(mask)
    max_z = 0
    x, y, z = 0, 0, 0
    for slice in range(len(mask)):
        min_i, min_j = len(mask[0]), len(mask[0][0])
        max_i, max_j = 0, 0
        for coloumn in range(len(mask[0])):
            for a in range(len(mask[0][0])):
                if mask[slice][coloumn][a] != 0:
                    if a < min_i:
                        min_i = a
                    if coloumn < min_j:
                        min_j = coloumn
                    if a > max_i:
                        max_i = a
                    if coloumn > max_j:
                        max_j = coloumn
                    if slice < min_z:
                        min_z = slice
                    if slice > max_z:
                        max_z = slice
        x_pr = max_i - min_i
        y_pr = max_j - min_j

        if x_pr > x:
            x = x_pr
        if y_pr > y:
            y = y_pr
    x = x * ps[0]
    y = y * ps[1]
    z = (max_z - min_z) * ps[2]
    return x, y, z


def get_voxel_volume(mask_coord: np.ndarray, sp: np.ndarray):
    return len(mask_coord) * sp[0] * sp[1] * sp[2]


def get_major_minor(mask_array: np.ndarray, ps: np.ndarray):
    labelledVoxelCoordinates = np.where(mask_array != 0)
    Np = len(labelledVoxelCoordinates[0])
    coordinates = np.array(labelledVoxelCoordinates, dtype='int').transpose((1, 0))
    physicalCoordinates = coordinates * ps[None, :]
    physicalCoordinates -= np.mean(physicalCoordinates, axis=0)  # Centered at 0
    physicalCoordinates /= np.sqrt(Np)
    covariance = np.dot(physicalCoordinates.T.copy(), physicalCoordinates)
    eigenValues = np.linalg.eigvals(covariance)
    eigenValues.sort()
    return np.sqrt(eigenValues[2]) * 4, np.sqrt(eigenValues[1]) * 4


if __name__ == "__main__":
    image_dicom = open_dicom(dicom_dir, dicom_start, dicom_stop)

    mask_nrrd = open_nrrd(mask_dir, mask_name)
    image_nrrd = open_nrrd(nrrd_dir, nrrd__image_name)

    spacing = np.array(image_nrrd.GetSpacing())

    mask_nrrd_array = get_array(mask_nrrd)
    image_nrrd_array = get_array(image_nrrd)

    img_orders = first_order(image_nrrd, mask_nrrd)
    img_shape = img_to_shape(image_nrrd, mask_nrrd)

    masked_image = get_masked_image(image_nrrd_array, mask_nrrd_array)

    mask_coord = get_coordinates(mask_nrrd_array)

    mean = get_mean(image_nrrd_array, mask_coord)
    sd = get_sd(image_nrrd_array, mask_coord, mean)
    median = get_median(image_nrrd_array, mask_coord)
    x, y, z = get_diams(mask_nrrd_array, spacing)
    minor, major = get_major_minor(mask_nrrd_array, spacing)
    voxel_volume = get_voxel_volume(mask_coord, spacing)

    print(f' mean = {mean}\n sd = {sd}\n median = {median}\n x = {x}\n y = {y}\n z = {z}\n major = {major}\n '
          f'minor = {minor}\n major = {major}\n voxel volume = {voxel_volume}')

    plot(masked_image[10])
