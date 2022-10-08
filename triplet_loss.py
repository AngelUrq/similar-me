# Adapted from https://github.com/rishiraj95/Face-Recognition-Triplet-Loss-on-Inception-v3

import tensorflow as tf

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.2) # alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

def calculate_distances(embeddings):
    embedding_matrix = tf.linalg.matmul(embeddings, tf.transpose(embeddings))
    norm_vector = tf.expand_dims(tf.linalg.diag_part(embedding_matrix), axis=1)

    distances = norm_vector - 2.0 * embedding_matrix + tf.transpose(norm_vector)
    distances = tf.maximum(distances, 0.0)
    
    mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)

    distances = distances + mask * 1e-16
    distances = tf.sqrt(distances) * (1.0 - mask)
    
    return distances

def get_triplet_mask(labels):
    batch_size = tf.shape(labels)[0]
    indices_equal = tf.cast(tf.eye(batch_size), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.logical_and(distinct_indices, valid_labels)
    
    return mask

def batch_triplet_loss(labels, embeddings, alpha):
    distances = calculate_distances(embeddings)
    
    triplet_loss = tf.expand_dims(distances, axis=2) - tf.expand_dims(distances, axis=1) + alpha
    
    valid_mask = get_triplet_mask(labels)
    valid_mask = tf.cast(valid_mask, tf.float32)
    
    triplet_loss = tf.multiply(triplet_loss, valid_mask)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(valid_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    
    return triplet_loss
