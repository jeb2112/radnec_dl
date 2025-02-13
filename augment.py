import numpy as np
import scipy
from scipy.ndimage import affine_transform

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.utils import (array_to_img,img_to_array)

# duplicate
def get_rotmat(r1,r2,r3):
    s1 = np.sin(r1)
    c1 = np.cos(r1)
    s2 = np.sin(r2)
    c2 = np.cos(r2)
    s3 = np.sin(r3)
    c3 = np.cos(r3)
    # tait-bryan X1Y2Z3
    Rxyz = np.reshape([c2*c3, -c2*s3, s2, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2],(3,3))
    # Rxyz = np.linalg.inv(Rxyz)
    # Rzyx = np.reshape([c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2, c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, -s2, c2*s3, c2*c3],(3,3))
    # Rzyx = np.linalg.inv(Rzyx)
    return Rxyz

# this was modified from transform_matrix_offset_center for rotation about arbitrary centre
# intepretation of x,y,z here is an arbitrary point, whereas the way apply_affine_transform 
# was coded the rotation point is the origin of the volume.
# should reconcile these
def get_affine(matrix, x, y, z):
    matrix = np.pad(matrix,((0,1),(0,1)))
    matrix[3,3] = 1
    o_x = float(x)
    o_y = float(y)
    o_z = float(z)
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0,0,1,o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0,0,1,-o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

# override from 2d
def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

# overrides 2d from affine_transformations.py
def apply_affine_transform(x, theta=0, phi=0, tx=0, ty=0, tz=0, shear=0, zx=1, zy=1, zz=1,
                        row_axis=0, col_axis=1, vol_axis=2, channel_axis=3,
                        fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation

    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                        'Install SciPy.')
    transform_matrix = None
    if theta != 0 or phi !=0:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        rotation_matrix = get_rotmat(phi,0,theta)
        transform_matrix = np.pad(rotation_matrix,((0,1),(0,1)))
        transform_matrix[3,3] = 1

    if tx != 0 or ty != 0 or tz != 0:
        shift_matrix = np.array([[1, 0, 0, tx],
                                [0, 1, 0, ty],
                                [0, 0, 1, tz],
                                [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if zx != 1 or zy != 1 or zz != 1:
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w, d = x.shape[row_axis], x.shape[col_axis], x.shape[vol_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w, d)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]

        channel_images = [affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


# subclass for a noise option and 3d
# **kwargs contains everything that can go to the parent init function
class RegressionAug(ImageDataGenerator):

    def __init__(self,noisevar=0.0,phi_rotation_range=2, vol_shift_range=0.1, **kwargs):
        super().__init__(self,
                        preprocessing_function=self.add_noise,
                        **kwargs)
        self.noisevar = noisevar
        self.phi_rotation_range = phi_rotation_range
        self.vol_shift_range = vol_shift_range
        self.interpolation_order = 1 # default 1

    def add_noise(self,a):
        n = np.random.standard_normal(a.shape) * self.noisevar
        return a+n

    # override for 3d
    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_vol_axis = self.col_axis # whether channels_first or last, the vol axis is tacked on here after the col axis

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        if self.phi_rotation_range:
            phi = np.random.uniform(
                -self.phi_rotation_range,
                self.phi_rotation_range)
        else:
            phi = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.vol_shift_range:
            try:  # 1-D array-like or int
                tz = np.random.choice(self.vol_shift_range)
                tz *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tz = np.random.uniform(-self.vol_shift_range,
                                       self.vol_shift_range)
            if np.max(self.vol_shift_range) < 1:
                tz *= img_shape[img_vol_axis]
        else:
            tz = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else: # no shear
            zx = np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                1)[0]
            zy = zx
            zz = zx

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(self.brightness_range[0],
                                           self.brightness_range[1])

        transform_parameters = {'theta': theta,
                                'phi': phi,
                                'tx': tx,
                                'ty': ty,
                                'tz': tz,
                                'zx': zx,
                                'zy': zy,
                                'zz': zz,
                                'brightness': brightness}

        return transform_parameters


    # override for 3d
    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_vol_axis = self.col_axis # whichever channel convention is, vol axis tacks on after col axis
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                                    transform_parameters.get('phi', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('tz', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   transform_parameters.get('zz', 1),
                                   row_axis=img_row_axis,
                                   col_axis=img_col_axis,
                                   vol_axis=img_vol_axis,
                                   channel_axis=img_channel_axis,
                                   fill_mode=self.fill_mode,
                                   cval=self.cval,
                                   order=self.interpolation_order)

        return x
