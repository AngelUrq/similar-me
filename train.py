import tensorflow as tf
from load_data import create_dataset, create_batch
from model import get_model
from triplet_loss import batch_triplet_loss

train_path = 'data/train'
test_path = 'data/test'

batch_size = 500
faces_per_identity = 10

model = get_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
alpha = 0.2

train_dataset = create_dataset(train_path)
test_dataset = create_dataset(test_path)

epochs = 300
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    
    train_batch = create_batch(train_dataset, batch_size=batch_size, faces_per_identity=faces_per_identity)

    for x_batch_train, y_batch_train in train_batch:
        with tf.GradientTape() as tape:
            embeddings = model(x_batch_train, training=True)
            loss_value = batch_triplet_loss(y_batch_train, embeddings, alpha)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        test_loss_value = 0
            
        print(
            "Training loss (for one batch) at step %d: %.4f"
            % (1, float(loss_value), float(test_loss_value))
        )
