// Naive split by spaces
const tokens = cleanedText.split(' ');

// Build frequency map
const freqMap = {};
for (const t of tokens) {
  if (!freqMap[t]) freqMap[t] = 0;
  freqMap[t]++;
}

// Sort tokens by frequency
const sortedTokens = Object.entries(freqMap).sort((a, b) => b[1] - a[1]);

// Build vocabulary (token -> id)
const vocab = {};
const reverseVocab = {};
sortedTokens.forEach(([token], index) => {
  vocab[token] = index;
  reverseVocab[index] = token;
});

console.log('Vocab size:', Object.keys(vocab).length);