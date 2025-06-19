import tensorflow as tf
import tensorflow_hub as hub

def load_img(path_to_img, image_size=(256, 256)):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = img[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def crop_center(image):
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

def stylize(content_path, style_path, output_path='stylized.jpg'):
    content_image = load_img(content_path, (384, 384))
    content_image = crop_center(content_image)

    style_image = load_img(style_path, (256, 256))
    style_image = crop_center(style_image)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    tf.keras.utils.save_img(output_path, stylized_image[0])
    return output_path
