# LSTM Text Generator

A TensorFlow/Keras LSTM text-generation project based on Shakespeare's works.

## Project Structure

```text
lstm-text-generator/
├── text_generator.py
├── shakespeare.txt
├── lstm_text_generator.keras   # created after training
└── README.md
```

## Requirements

- Python 3.10+
- TensorFlow
- NumPy

Install:

```bash
pip install tensorflow numpy
```

## Dataset

The dataset is based on Project Gutenberg's **The Complete Works of William Shakespeare**, eBook #100:
https://www.gutenberg.org/ebooks/100

Download the Plain Text version and save it as `shakespeare.txt` in this folder.

## Run

```bash
python text_generator.py
```

The program:
1. Loads and cleans the text.
2. Tokenizes words.
3. Creates next-word input/output sequences.
4. Splits data into training and validation sets.
5. Trains an Embedding + 2-layer LSTM + Softmax model.
6. Uses EarlyStopping to reduce overfitting.
7. Saves the trained model as `lstm_text_generator.keras`.
8. Generates text from multiple seed phrases.

## Sample Seeds

- `to be or not to be`
- `the king said`
- `love is`

## Bonus Experiment

Change `SEQ_LENGTH` from `20` to `30` or `40`, or add another LSTM layer, then compare validation loss and generated-text quality.

## Task Coverage

- Data preprocessing: Yes
- Tokenization: Yes
- Input/output sequences: Yes
- LSTM architecture: Yes
- Training/validation split: Yes
- Early stopping: Yes
- Text generation: Yes
- Model saving: Yes
