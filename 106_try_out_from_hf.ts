import { pipeline } from '@xenova/transformers';

(async () => {
  // Load text-generation pipeline pointing to your scratch model
  const generator = await pipeline(
    'text-generation',
    'guerrerocarlos/my-gpt2-from-scratch'
  );

  // Generate text
  const output = await generator("Hello, I'm trying out", {
    max_length: 50,
    num_return_sequences: 1
  });

  console.log(output);
  // => [ { generated_text: "... some text ..." } ]
})();