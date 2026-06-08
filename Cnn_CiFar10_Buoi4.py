from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Kich thuoc tap du train: {x_train.shape}")
print(f"Kich thuoc tap test: {x_test.shape}")


plt.figure(figsize=(10,4))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

x_train, x_test = x_train/255.0, x_test/255.0

class_name = ['airphone','automobile','bird','cat','deer','dog','frog','horse',
              'ship','truck']

model= models.Sequential([
        layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu',
            input_shape=(32,32,3)
            ),
        layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu'
        ),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.45),

        layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu',
            input_shape=(32,32,3)
            ),
        layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu'
        ),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.45),

        layers.Flatten(),

        layers.Dense(units=512,activation='relu'),
        layers.Dropout(0.45),
        layers.Dense(units=10,activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history =model.fit(
    x_train,y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_test,y_test)
)


test_loss, test_acc = model.evaluate(x_test,y_test,verbose=1)

print(f'Test accuracy: {test_acc:.4f}')

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Accuracy',marker='o',color='b')
plt.plot(history.history['val_accuracy'],label='validation Accuracy',marker='o',color='r')
plt.title("Đo chinh xac qua cac epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)


plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train_loss',marker='o',color='b')
plt.plot(history.history['val_loss'],label='Validation Loss',marker='o',color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Do mat mat qua cac epoch')
plt.legend(loc='upper right')
plt.grid(True)


plt.tight_layout()
plt.show()

