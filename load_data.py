import tensorflow as tf
import os

def list_files_in_folder(filepath):
    filepath = tf.strings.join([filepath, f'{os.path.sep}*.jpg'])
    
    return tf.data.Dataset.list_files(filepath, shuffle=True)

def get_label(filepath):
    return tf.strings.split(filepath, os.path.sep)[2]

def process_image(filepath):
    label = get_label(filepath)
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    
    return tf.reshape(img, (128, 128, 3)) / 255.0, label

def create_dataset(path):
    dataset = tf.data.Dataset.list_files(f'{path}/*', shuffle=True)
    dataset = dataset.map(list_files_in_folder)
    
    return dataset
    
def create_batch(dataset, batch_size, faces_per_identity):
    identities = batch_size // faces_per_identity
    first = True
    
    identity_dataset = dataset.take(identities)

    for identity in identity_dataset:
        if first:
            faces = identity.take(faces_per_identity)
            first = False
        else:
            faces = faces.concatenate(identity.take(faces_per_identity))

    faces = faces.map(process_image)
    batch = faces.batch(batch_size)
    
    return batch
