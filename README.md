# 🤖 BERT Masked Word Prediction with Attention Visualization

This project demonstrates how a **Masked Language Model (MLM)**, specifically **BERT**, predicts masked words in natural language text and visualizes its **self-attention mechanisms**.

Built using the **Hugging Face Transformers** library, the project highlights how attention heads in BERT "focus" on different parts of a sentence to interpret meaning.

---

## 📌 Overview

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model trained using masked language modeling (MLM), where a word in a sentence is replaced with a `[MASK]` token and the model is tasked with predicting it.

This project does two things:

1. **Predicts the masked word** in a given sentence.
2. **Generates attention diagrams** for each of the 144 attention heads (12 layers × 12 heads) to help us interpret how BERT understands language.

---

## 🧪 Example Usage

```bash
$ python mask.py
Text: Then I picked up a [MASK] from the table.
Then I picked up a book from the table.
Then I picked up a bottle from the table.
Then I picked up a plate from the table.
```

After prediction, it generates 144 attention diagrams showing how each attention head distributed its focus across tokens in the sentence.

---

## 🛠 Implementation Details

### ✅ Functions Implemented

- **`get_mask_token_index(mask_token_id, inputs)`**  
  Returns the index of the `[MASK]` token from the input tensor.

- **`get_color_for_attention_score(score)`**  
  Converts an attention score (0–1) into a grayscale RGB value for visualization.

- **`visualize_attentions(tokens, attentions)`**  
  Iterates over all 144 attention heads and generates heatmaps using the attention scores.

### 🎨 Attention Diagrams

- Each diagram represents how one attention head distributes its attention across tokens.
- Brighter cells = higher attention.
- Generated using PIL and saved as image files.

---

## 🔬 Language Insight

By visualizing attention heads, we can interpret the relationships BERT learns, such as:
- Which words attend to verbs
- Pronoun resolution
- Prepositional phrase attachment
- Modifiers and dependencies

*Analysis of two such heads is included in `analysis.md`.*

---

## 🧠 Tech Stack

- Python 3.12
- Hugging Face Transformers
- TensorFlow
- NumPy
- Pillow (for image creation)

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run the script:**
   ```bash
   python mask.py
   ```

3. **Enter a sentence** with `[MASK]` in it.

4. **View predictions and attention diagrams.**

---

## 📁 File Structure

```
attention/
├── mask.py           # Main script for masked word prediction & visualization
├── analysis.md       # Write-up interpreting two specific attention heads
├── requirements.txt  # Python dependencies
├── *.png             # Output images of attention heads
```

---

## 🧠 What I Learned

- How BERT uses self-attention to predict missing words
- Interpreting multi-headed attention in transformers
- Visualizing attention scores as heatmaps
- Hands-on experience with Hugging Face's Transformers library

---

## ✅ Bonus

You can also analyze your output by examining attention heads across various layers to discover language patterns BERT has learned!

```bash
check50 ai50/projects/2024/x/attention
style50 mask.py
```

---
