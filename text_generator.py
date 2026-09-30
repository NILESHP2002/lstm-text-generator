import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

DATASET = "shakespeare.txt"
MODEL_PATH = "lstm_text_generator.keras"
SEQ_LENGTH = 20


def load_and_clean_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_data(text):
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts([text])
    tokens = tokenizer.texts_to_sequences([text])[0]
    sequences = [tokens[i-SEQ_LENGTH:i+1] for i in range(SEQ_LENGTH, len(tokens))]
    sequences = np.array(sequences)
    X = sequences[:, :-1]
    y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=len(tokenizer.word_index) + 1)
    split = int(len(X) * 0.9)
    return tokenizer, X[:split], X[split:], y[:split], y[split:]


def build_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=SEQ_LENGTH),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def generate_text(model, tokenizer, seed, words=50):
    result = seed.lower()
    for _ in range(words):
        seq = tokenizer.texts_to_sequences([result])[0][-SEQ_LENGTH:]
        seq = pad_sequences([seq], maxlen=SEQ_LENGTH)
        probs = model.predict(seq, verbose=0)[0]
        next_id = int(np.argmax(probs))
        next_word = tokenizer.index_word.get(next_id)
        if not next_word or next_word == "<unk>":
            break
        result += " " + next_word
    return result


def main():
    text = load_and_clean_text(DATASET)
    tokenizer, X_train, X_val, y_train, y_val = prepare_data(text)
    model = build_model(len(tokenizer.word_index) + 1)
    model.summary()

    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20,
              batch_size=128, callbacks=callbacks)
    model.save(MODEL_PATH)

    print("\nModel saved:", MODEL_PATH)
    for seed in ["to be or not to be", "the king said", "love is"]:
        print("\nSeed:", seed)
        print("Generated:", generate_text(model, tokenizer, seed, 30))


if __name__ == "__main__":
    main()
