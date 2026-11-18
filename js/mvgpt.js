// mvGPT in the browser, with Transformers.js: the two models, the prompt the fine-tuned one was trained
// on, and writing a story a token at a time. The home page's chat (TryModel.astro) and /try/
// (Chat.astro) both import it. Nothing downloads until the first story: then the library, from
// jsDelivr, and the 8-bit model, from the Hugging Face Hub, which the browser keeps for the next visit.
const LIBRARY = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
const REPO = 'nenuadrian/mvGPT';

/** Revisions of the model repo: each one's context window, and whether it was fine-tuned on instructions. */
export const MODELS = {
  sft: { revision: 'ep003-sft', context: 256, instructions: true },
  big: { revision: 'ep012-pretrain', context: 512, instructions: false },
};

/** mvGPT's detokenize: punctuation goes against the word before it. */
export function detokenize(tokens) {
  return tokens
    .join(' ')
    .replace(/ ([.,!?;:]\S*)/g, '$1')
    .replace(/(\w) ' (\w)/g, "$1'$2")
    .replace(/(^|\s)" /g, '$1"');
}

/** A list of short words is the words to use, and anything else the story's summary. */
export function words(text) {
  const parts = text
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);
  const list = parts.length > 1 && parts.every((part) => part.split(/\s+/).length <= 2) && !/[.!?]$/.test(text);
  return list ? parts : null;
}

/** The instruction's lines, in the order the model was fine-tuned on. */
export function instruction(text, features) {
  const list = words(text);
  const lines = features.length ? [['Features', features.join(', ')]] : [];
  lines.push(list ? ['Words', list.join(', ')] : ['Summary', /[.!?]$/.test(text) ? text : `${text}.`]);
  return [...lines, ['Story', '']];
}

/** The instruction as the model reads it. */
export function promptOf(lines) {
  return lines.map(([name, value]) => (value ? `${name}: ${value}` : `${name}:`)).join('\n');
}

/** "145 tokens · 0.7 s · 203 tokens/s" */
export function stats({ count, seconds }) {
  return count ? `${count} tokens · ${seconds.toFixed(1)} s · ${Math.round(count / seconds)} tokens/s` : '';
}

// The library, loaded on the first story, and each model on its own first story.
let library = null;
const writers = {};

/**
 * Carries on from the prompt with the named model. `onStatus` hears what's downloading, and '' once
 * the model is ready; `onText` gets the story so far after each token; aborting `signal` stops it
 * after the token it's on. Resolves to how many tokens it wrote and how long that took.
 */
export async function write(name, prompt, { onStatus = () => {}, onText = () => {}, signal } = {}) {
  const model = MODELS[name];
  if (!library) {
    onStatus('downloading the library…');
    library = await import(LIBRARY);
  }
  if (!writers[name]) {
    writers[name] = await library.pipeline('text-generation', REPO, {
      revision: model.revision,
      dtype: 'q8',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file?.endsWith('.onnx')) onStatus(`downloading mvGPT ${Math.round(p.progress ?? 0)}%`);
      },
    });
  }
  const writer = writers[name];
  onStatus('');
  const ids = [];
  let count = 0;
  const streamer = new library.TextStreamer(writer.tokenizer, {
    skip_prompt: true,
    callback_function: () => {},
    token_callback_function: (tokens) => {
      ids.push(...tokens);
      count += 1;
      // The word-level tokenizer drops whitespace, which mvGPT's detokenize puts back; BPE keeps every
      // byte, spaces and new lines included, so its tokens decode as they are.
      onText(
        model.instructions
          ? detokenize(ids.map((id) => writer.tokenizer.decode([id], { skip_special_tokens: true })).filter(Boolean))
          : writer.tokenizer.decode(ids, { skip_special_tokens: true }),
      );
    },
  });
  // The context window holds the prompt and the story together.
  const max_new_tokens = Math.min(200, model.context - writer.tokenizer.encode(prompt).length);
  const options = { max_new_tokens, do_sample: true, temperature: 0.8, top_k: 50, streamer };
  if (signal) {
    const stopper = new library.InterruptableStoppingCriteria();
    if (signal.aborted) stopper.interrupt();
    signal.addEventListener('abort', () => stopper.interrupt(), { once: true });
    options.stopping_criteria = stopper;
  }
  const start = performance.now();
  await writer(prompt, options);
  return { count, seconds: (performance.now() - start) / 1000 };
}
