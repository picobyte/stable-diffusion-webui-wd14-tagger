""" Credits to SmilingWolf """

import tensorflow as tf
try:
    import tensorflow_io as tfio  # pylint: disable=import-error
except ImportError:
    tfio = None

def is_webp(contents):
    """Checks if the image is a webp image"""
    riff_header = tf.strings.substr(contents, 0, 4)
    webp_header = tf.strings.substr(contents, 8, 4)

    is_riff = riff_header == b"RIFF"
    is_fourcc_webp = webp_header == b"WEBP"
    return is_riff and is_fourcc_webp


class DataGenerator:
    """ Data generator for the dataset """
    def __init__(self, file_list, target_height, target_width, batch_size):
        self.file_list = file_list
        self.target_width = target_width
        self.target_height = target_height
        self.batch_size = batch_size

    def read_image(self, filename):
        image_bytes = tf.io.read_file(filename)
        return filename, image_bytes

    def parse_single_image(self, filename, image_bytes):
        """ Parses a single image """
        if is_webp(image_bytes):
            image = tfio.image.decode_webp(image_bytes)
        else:
            image = tf.io.decode_image(
                image_bytes, channels=0, dtype=tf.uint8,
                expand_animations=False
            )

        # Black and white image
        if tf.shape(image)[2] == 1:
            image = tf.repeat(image, 3, axis=-1)

        # Black and white image with alpha
        elif tf.shape(image)[2] == 2:
            image, mask = tf.unstack(image, num=2, axis=-1)
            mask = tf.expand_dims(mask, axis=-1)
            image = tf.expand_dims(image, axis=-1)
            image = tf.repeat(image, 3, axis=-1)
            image = tf.concat([image, mask], -1)

        # Alpha to white
        if tf.shape(image)[2] == 4:
            alpha_mask = image[:, :, 3]
            alpha_mask = tf.cast(alpha_mask, tf.float32) / 255
            alpha_mask = tf.repeat(tf.expand_dims(alpha_mask, -1), 4, axis=-1)

            matte = tf.ones_like(image, dtype=tf.uint8) * [255, 255, 255, 255]

            weighted_matte = tf.cast(matte, dtype=alpha_mask.dtype) * (1 - alpha_mask)  # noqa: E501
            weighted_image = tf.cast(image, dtype=alpha_mask.dtype) * alpha_mask  # noqa: E501
            image = weighted_image + weighted_matte

            # Remove alpha channel
            image = tf.cast(image, dtype=tf.uint8)[:, :, :-1]

        # Pillow/Tensorflow RGB -> OpenCV BGR
        image = image[:, :, ::-1]
        return filename, image

    def resize_single_image(self, filename, image):
        """ Resizes a single image """
        height, width, _ = tf.unstack(tf.shape(image))

        if height <= self.target_height and width <= self.target_width:
            return filename, image

        image = tf.image.resize(
            image,
            (self.target_height, self.target_width),
            method=tf.image.ResizeMethod.AREA,
            preserve_aspect_ratio=True,
        )
        image = tf.cast(tf.math.round(image), dtype=tf.uint8)
        return filename, image

    def pad_single_image(self, filename, image):
        """ Pads a single image """
        height, width, _ = tf.unstack(tf.shape(image))

        float_h = tf.cast(height, dtype=tf.float32)
        float_w = tf.cast(width, dtype=tf.float32)
        float_target_h = tf.cast(self.target_height, dtype=tf.float32)
        float_target_w = tf.cast(self.target_width, dtype=tf.float32)

        padding_top = tf.cast((float_target_h - float_h) / 2, dtype=tf.int32)
        padding_right = tf.cast((float_target_w - float_w) / 2, dtype=tf.int32)
        padding_bottom = self.target_height - padding_top - height
        padding_left = self.target_width - padding_right - width

        padding = [[padding_top, padding_bottom],
                   [padding_right, padding_left], [0, 0]]
        image = tf.pad(image, padding, mode="CONSTANT", constant_values=255)
        return filename, image

    def gen_ds(self):
        """ Generates the dataset """
        if tfio is None:
            print("Tensorflow IO is not installed, try\n"
                  "`pip install tensorflow_io' or use another interrogator")
            return []
        images_list = tf.data.Dataset.from_tensor_slices(self.file_list)

        images_data = images_list.map(
            self.read_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        images_data = images_data.map(
            self.parse_single_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        images_data = images_data.map(
            self.resize_single_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        images_data = images_data.map(
            self.pad_single_image, num_parallel_calls=tf.data.AUTOTUNE
        )

        images_list = images_data.batch(
            self.batch_size, drop_remainder=False,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        images_list = images_list.prefetch(tf.data.AUTOTUNE)
        return images_list
