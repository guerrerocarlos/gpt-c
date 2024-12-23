const tf = require('@tensorflow/tfjs-node');

function createTinyTransformer(vocabSize, embeddingDim, seqLen, numHeads, feedForwardDim) {
  const input = tf.input({ shape: [seqLen], dtype: 'int32' });
  
  // 1. Token embedding
  const embeddingLayer = tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingDim
  }).apply(input);
  
  // 2. Positional encoding (naive approach: treat position as a trainable embedding)
  const positionInput = tf.range(0, seqLen, 1);
  const positionEmbeddingLayer = tf.layers.embedding({
    inputDim: seqLen,
    outputDim: embeddingDim
  }).apply(positionInput);
  
  // We add position embedding to token embedding
  let x = tf.layers.add().apply([embeddingLayer, positionEmbeddingLayer]);
  
  // 3. Self-attention layer
  //    A minimal multi-head attention in TF.js
  const multiHeadAttention = tf.layers.multiHeadAttention({
    numHeads: numHeads,
    keyDim: embeddingDim,  // dimension per head
    // in real models we have more configs like dropout, etc.
  }).apply([x, x, x]);
  
  // Residual connection + layer norm
  x = tf.layers.add().apply([x, multiHeadAttention]);
  x = tf.layers.layerNormalization().apply(x);
  
  // 4. Feed-forward
  let ff = tf.layers.dense({ units: feedForwardDim, activation: 'relu' }).apply(x);
  ff = tf.layers.dense({ units: embeddingDim }).apply(ff);
  
  // Residual + layer norm
  x = tf.layers.add().apply([x, ff]);
  x = tf.layers.layerNormalization().apply(x);
  
  // 5. Flatten and final classifier
  //    We only want the last token's output for next-token prediction
  //    This means we can take x[:, -1, :] and then do a dense to vocab
  const lastToken = tf.layers.lambda({
    func: (tensor) => tensor.gather([seqLen - 1], 1) // gather last position
  }).apply(x);
  
  const logits = tf.layers.dense({
    units: vocabSize,
    activation: 'linear'
  }).apply(lastToken);
  
  // Create model
  const model = tf.model({ inputs: input, outputs: logits });
  return model;
}

// Hyperparameters (tune for your case)
const VOCAB_SIZE = Object.keys(vocab).length;
const EMBEDDING_DIM = 32;
const SEQ_LENGTH = SEQ_LEN;
const NUM_HEADS = 2;
const FEED_FORWARD_DIM = 64;

const tinyTransformer = createTinyTransformer(
  VOCAB_SIZE,
  EMBEDDING_DIM,
  SEQ_LENGTH,
  NUM_HEADS,
  FEED_FORWARD_DIM
);

tinyTransformer.summary();
