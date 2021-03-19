# Abstract patterns neural generator

This script uses pre-trained DCGAN (Deep Convolutional Generative Adversarial Network) to generate patterns resembling handwritten digits (the network was trained on MNIST dataset) and assembling them into a looped mp4 clip.

<details>
  <summary>Original structure of DCGAN used in this project</summary>
  
  ```python
  generator = M.Sequential([
      L.Dense(128*7*7, activation="relu"),
      L.Reshape((7, 7, 128)),
      L.UpSampling2D((2, 2)),    
      L.Conv2D(128, (3, 3), padding="same"),
      L.BatchNormalization(momentum=0.8),
      L.ReLU(),    
      L.UpSampling2D((2, 2)),    
      L.Conv2D(64, (3, 3), padding="same"),
      L.BatchNormalization(momentum=0.8),
      L.ReLU(),    
      L.Conv2D(1, (3, 3), padding="same", activation='tanh'),
  ])

  discriminator = M.Sequential([
      L.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
      L.LeakyReLU(0.2),
      L.Dropout(0.25),    
      L.Conv2D(64, kernel_size=3, strides=(2, 2), padding="same"),
      L.ZeroPadding2D(padding=((0, 1), (0, 1))),
      L.BatchNormalization(momentum=0.8),
      L.LeakyReLU(alpha=0.2),
      L.Dropout(0.25),
      L.Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"),
      L.BatchNormalization(momentum=0.8),
      L.LeakyReLU(alpha=0.2),
      L.Dropout(0.25),
      L.Conv2D(256, kernel_size=3, strides=(1, 1), padding="same"),
      L.BatchNormalization(momentum=0.8),
      L.LeakyReLU(alpha=0.2),
      L.Dropout(0.25),
      L.Flatten(),
      L.Dense(1, activation=None),
  ])
  ```

</details>

<details>
  <summary>Training parameters and training process</summary>
  
  ```python
  def plot_digits(samples):
      fig = plt.figure(figsize=(10, 10))
      num = samples.shape[0]
      for j in range(num):
          ax = fig.add_subplot(8, 8, j+1)
          ax.imshow(samples[j, ...].reshape(28, 28), cmap='gray')
          plt.xticks([]), plt.yticks([])
      plt.show()
      
  # Load MNIST dataset
  (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
  train_x = (train_x.reshape(-1, 28*28).astype(np.float32) - 127.5) / 127.5

  INPUT_DIM = 100
  NUM_EPOCHS = 2
  HALF_BATCH_SIZE = 16
  BATCH_SIZE = HALF_BATCH_SIZE * 2
  LEARNING_RATE = 0.0002

  train_ds = tf.data.Dataset.from_tensor_slices(train_x.reshape(-1, 28, 28, 1))
  train_ds = train_ds.shuffle(buffer_size=train_x.shape[0])
  train_ds = train_ds.repeat(NUM_EPOCHS)
  train_ds = train_ds.batch(HALF_BATCH_SIZE, drop_remainder=True)

  optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

  for step, true_images in enumerate(train_ds):

      # Train Discriminator

      noise = np.random.normal(0, 1, (HALF_BATCH_SIZE, INPUT_DIM)).astype(np.float32)
      syntetic_images = generator.predict(noise)
      x_combined = np.concatenate((
          true_images, 
          syntetic_images))
      y_combined = np.concatenate((
          np.ones((HALF_BATCH_SIZE, 1)), 
          np.zeros((HALF_BATCH_SIZE, 1))))

      with tf.GradientTape() as tape:
          logits = discriminator(x_combined, training=True)
          d_loss_value = tf.compat.v1.losses.sigmoid_cross_entropy(y_combined, logits)
      grads = tape.gradient(d_loss_value, discriminator.trainable_variables)
      optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

      # Train Generator

      noise = np.random.normal(0, 1, (BATCH_SIZE, INPUT_DIM)).astype(np.float32)
      y_mislabled = np.ones((BATCH_SIZE, 1))

      with tf.GradientTape() as tape:
          syntetic = generator(noise, training=True)
          logits = discriminator(syntetic, training=False)
          g_loss_value = tf.compat.v1.losses.sigmoid_cross_entropy(y_mislabled, logits)
      grads = tape.gradient(g_loss_value, generator.trainable_variables)
      optimizer.apply_gradients(zip(grads, generator.trainable_variables))

      # Check intermediate results

      if step % 200 == 0:
          print("[Step %2d] D Loss: %.4f; G Loss: %.4f" % (
              step, d_loss_value.numpy(), g_loss_value.numpy()))
          noise = np.random.normal(0, 1, (8, INPUT_DIM)).astype(np.float32)
          syntetic_images = generator.predict(noise)
          plot_digits(syntetic_images)
  ```
</details>

__Please note that this network wasn't tuned and trained perfectly on purpose.__ This whole project was made to generate abstract art resembling handwritten digits.

In ```create_images()``` function you can adjust the number of generated images, in ```create_video()``` function the frame rate of the clip can be adjusted. These two parameters will define the overall length of the resulting clip.

The script uses the following libraries:
- tensorflow;
- os;
- cv2;
- numpy;
- glob;
- tqdm;
- matplotlib.

The code was written according to PEP8.
