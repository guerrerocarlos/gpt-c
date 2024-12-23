// Convert entire dataset into numeric IDs
const tokenIds = tokens.map(t => vocab[t]);

// For demonstration, let's make small sequences
// e.g. sample length = 10
const SEQ_LEN = 10;
let sequences = [];
let targets = [];

for (let i = 0; i < tokenIds.length - SEQ_LEN; i++) {
  const inputSeq = tokenIds.slice(i, i + SEQ_LEN);
  const targetToken = tokenIds[i + SEQ_LEN];
  
  sequences.push(inputSeq);
  targets.push(targetToken);
}

// Now we have training data in "sequences" (X) and "targets" (y)