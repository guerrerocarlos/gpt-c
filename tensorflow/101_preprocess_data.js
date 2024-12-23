const fs = require('fs');

// Load your dataset (ensure it's not huge for demonstration).
const rawText = fs.readFileSync('./dataset.txt', 'utf8');

// Basic cleaning (optional or can be more sophisticated)
let cleanedText = rawText
  .toLowerCase()
  .replace(/[^a-z0-9.?!\s]/g, '') // remove punctuation except . ? !
  .replace(/\s+/g, ' ')          // multiple spaces -> single space
  .trim();

console.log(cleanedText.slice(0, 200));  // preview