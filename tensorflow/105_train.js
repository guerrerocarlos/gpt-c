// Convert to Tensors
const xTrain = tf.tensor2d(sequences, [sequences.length, SEQ_LEN], 'int32');
const yTrain = tf.tensor1d(targets, 'int32');

tinyTransformer.compile({
  optimizer: tf.train.adam(),
  loss: tf.losses.sparseCategoricalCrossentropy, // since targets are int
  metrics: ['accuracy'],
});

// Let’s do a small training loop for demonstration
(async () => {
  const BATCH_SIZE = 32;
  const EPOCHS = 2; // increase for better results
  
  await tinyTransformer.fit(xTrain, yTrain, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} / ${EPOCHS} | loss: ${logs.loss.toFixed(4)} | acc: ${logs.acc.toFixed(4)}`
        );
      }
    }
  });
})();