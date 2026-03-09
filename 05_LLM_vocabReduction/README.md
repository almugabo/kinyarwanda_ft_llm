
# Vocabulary Reduction & Finetuning LLM to a new language 

New research shows that one can adapt an LLM to another language by first reducing the vocabulary of the base LLM (by training a new tokenizer) and using it in the model 


Here we provide a proof of concept for **Gemma 3 270M** to a Kinyarwanda using vocabulary reduction , this saves compute and memory while maintaining task performance.


## Step 1: Train a Bilingual Tokenizer

To ensure the model understands both English and Kinyarwanda, you must train the tokenizer tailored to your specific languages.

You need to modify the data iterator in your script (from File 2) to yield a mix of English and Kinyarwanda text. By feeding the BPE trainer a mixed stream, it will naturally learn the optimal subwords for both languages.


## Step 2: Remap and Initialize Embeddings

Once your bilingual tokenizer is saved, you must replace the tokenizer in the original model. You cannot simply drop tokens; you must build a new embedding matrix.

Following the logic in File 3, you will create a new embedding matrix (E1) sized to your new vocabulary.

  1. **Keep Surviving Tokens**: Iterate through your new vocabulary. If a token exists in the original Gemma tokenizer, copy its exact embedding over to the new matrix. Because you trained on English, many standard English words will naturally survive and retain their original weights.

  2. **Initialize New Tokens**: For genuinely new Kinyarwanda tokens, initialize their embeddings by averaging the embeddings of their constituent pieces from the original tokenizer. You can use the init_avg_old function from File 3, which weights each piece's embedding by its string length to bias toward longer subwords. If no valid pieces contribute, it falls back to a random vector.

## Step 3: Resize and Tie the Model Weights

With your new embedding matrix (E1) ready, you must inject it into the Gemma model and update the configuration so it doesn't break during generation.

- Resize the token embeddings in the model (model.resize_token_embeddings(V_new)).

- Replace the original weights with your new matrix (model.get_input_embeddings().weight[:] = E1).

- Crucial Step: For Gemma 3 270M, roughly 63% of the parameters are in the embedding table. Ensure the LM head is tied to these new embeddings by setting model.config.tie_word_embeddings = True and calling model.tie_weights().

- Update the eos_token_id, bos_token_id, and pad_token_id in the model's configuration to match your new tokenizer. If configuration is wrong, you can get failures that are hard to diagnose.

## Step 4: Continuous Pre-training

After swapping the tokenizer and embeddings, the model's parameters are out of sync with the new tokenization distribution, so the model will be unstable.

Before fine-tuning on specific tasks, you must do a continuous pre-training phase:

- Use a large unlabeled corpus in both English and Kinyarwanda.

- Train it using standard causal language modeling (next-token prediction) without any chat templates.

- The goal here isn't task learning, but rather to realign the network with the new tokenization. For a model as small as Gemma 3 270M, a full end-to-end retrain of the parameters is best by a wide margin.

-----------------

## Sources 


Efficient Vocabulary Reduction for Small Language Models  https://aclanthology.org/2025.coling-industry.64.pdf

Substack : Benjamin Marie Shrink LLMs with Vocabulary Reduction_ From Gemma 3 270M to 141M





