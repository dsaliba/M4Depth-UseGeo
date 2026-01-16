import tensorflow as tf
import numpy as np
from .generic import DataLoaderGeneric


# Dataloader for UseGeo dataset, generate bindings using accompanying scripts found in /scripts as well as the npy converter 
class DataLoaderUseGeo(DataLoaderGeneric):
    """
    UseGeo dataloader:
    - Consumes rows from a CSV with header:
      RGB_im, depth, f_x, f_y, c_x, c_y, rot_w, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z
    - Outputs the standard M4Depth sample dict:
      {RGB_im, depth, camera{f,c}, rot, trans, new_traj}
    - Resizes RGB and depth to 384x384 with static shapes.
    """

    def __init__(self):
        super(DataLoaderUseGeo, self).__init__('usegeo')

        
        self.depth_type = "map"

        
        self.out_h = 384
        self.out_w = 384

    # I was struggling with tiff files so I convert to npy first
    def _load_npy_py(self, path):
        arr = np.load(path.decode()).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]  # [H, W] -> [H, W, 1]
        return arr

    #read csv from split generation
    def _decode_samples(self, sample):
        """
        `sample` keys come from the CSV header via DataLoaderGeneric.
        Expected keys:
          - "RGB_im": path to RGB image (JPEG/PNG)
          - "depth": path to depth .npy
          - "f_x", "f_y", "c_x", "c_y": intrinsics in pixels
          - "rot_w", "rot_x", "rot_y", "rot_z"
          - "trans_x", "trans_y", "trans_z"
        """

        # --- RGB ---
        rgb_bytes = tf.io.read_file(sample["RGB_im"])
        rgb_raw = tf.image.decode_jpeg(rgb_bytes, channels=3)
        rgb_raw = tf.image.convert_image_dtype(rgb_raw, tf.float32)
        rgb = tf.image.resize(rgb_raw, [self.out_h, self.out_w])
        rgb = tf.ensure_shape(rgb, [self.out_h, self.out_w, 3])

        # --- Depth (.npy) --- (NOT TIFF, TIFF FILES WERE NOT PLAYING WELL WITH THIS VERSION OF TF)
        depth_raw = tf.numpy_function(self._load_npy_py, [sample["depth"]], tf.float32)
        depth_raw.set_shape([None, None, 1])
        depth = tf.image.resize(depth_raw, [self.out_h, self.out_w])
        depth = tf.ensure_shape(depth, [self.out_h, self.out_w, 1])

        # --- Camera intrinsics conversions needed if resizing
        fx = tf.cast(sample["f_x"], tf.float32)
        fy = tf.cast(sample["f_y"], tf.float32)
        cx = tf.cast(sample["c_x"], tf.float32)
        cy = tf.cast(sample["c_y"], tf.float32)

        camera = {
            "f": tf.stack([fx, fy]), 
            "c": tf.stack([cx, cy]),
        }

        # --- Pose ---
        rot = tf.stack(
            [
                tf.cast(sample["rot_w"], tf.float32),
                tf.cast(sample["rot_x"], tf.float32),
                tf.cast(sample["rot_y"], tf.float32),
                tf.cast(sample["rot_z"], tf.float32),
            ],
            axis=0,
        )
        trans = tf.stack(
            [
                tf.cast(sample["trans_x"], tf.float32),
                tf.cast(sample["trans_y"], tf.float32),
                tf.cast(sample["trans_z"], tf.float32),
            ],
            axis=0,
        )

        out = {
            "RGB_im": rgb,
            "depth": depth,
            "camera": camera,
            "rot": rot,
            "trans": trans,
            "new_traj": tf.constant(False),
        }

        return out

    
    def _perform_augmentation(self):
        pass
