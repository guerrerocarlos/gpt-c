function generateText(model, prompt, length = 20) {
  let inputIds = prompt.map(t => vocab[t] || 0); // fallback to 0 if not in vocab
  
  for (let i = 0; i < length; i++) {
    // 1. If inputIds is shorter than SEQ_LEN, pad left with some pad token or 0
    let sequence = inputIds.slice(-SEQ_LEN);
    if (sequence.length < SEQ_LEN) {
      sequence = new Array(SEQ_LEN - sequence.length).fill(0).concat(sequence);
    }
    // 2. Convert to tensor
    const inputTensor = tf.tensor2d([sequence], [1, SEQ_LEN], 'int32');
    // 3. Predict
    const logits = model.predict(inputTensor);
    const probabilities = logits.softmax().dataSync(); // shape [vocab_size]
    // 4. Sample from distribution (or take argmax)
    const nextTokenId = sampleFromDistribution(probabilities); 
    // 5. Append token
    inputIds.push(nextTokenId);
  }
  
  // Convert numeric IDs back to text
  const outputTokens = inputIds.map(id => reverseVocab[id]);
  return outputTokens.join(' ');
}

// Simple function to sample from probability distribution
function sampleFromDistribution(probabilities) {
  let sum = 0;
  const r = Math.random();
  for (let i = 0; i < probabilities.length; i++) {
    sum += probabilities[i];
    if (r <= sum) return i;
  }
  return probabilities.length - 1; // fallback
}

// Example usage:
const prompt = ['the', 'meaning', 'of']; // some words in your vocabulary
const generated = generateText(tinyTransformer, prompt);
console.log('Generated text:', generated);