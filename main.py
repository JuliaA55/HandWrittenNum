import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🔁 Навчання моделі...")
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Точність на тестових даних: {test_acc:.4f}")

model.save("mnist_digit_model.h5")
print("💾 Модель збережена як mnist_digit_model.h5")

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.title('Навчання моделі')
plt.show()
plt.savefig("training_accuracy.png")
print("📈 Графік точності збережено як training_accuracy.png")


def predict_digit(image_path):
    print("🔍 Шлях до зображення:", os.path.abspath(image_path))
    if not os.path.exists(image_path):
        print(f"❌ Файл '{image_path}' не знайдено.")
        return

    model = tf.keras.models.load_model("mnist_digit_model.h5")

    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\n🔢 Передбачена цифра: {predicted_class} (впевненість: {confidence:.2%})")


predict_digit("images/image.png")
