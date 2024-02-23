import tensorflow as tf


def get_train_data(batch_size=64, image_size=(256, 256)):

    train_data = tf.keras.utils.image_dataset_from_directory(
        './train_data',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        interpolation='bilinear',
    )

    train_data = train_data.take(train_data.__len__() - 1)
    return train_data

def get_valid_data(batch_size=64, image_size=(256, 256)):

    valid_data = tf.keras.utils.image_dataset_from_directory(
        './valid_data',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        interpolation='bilinear',
    )

    valid_data = valid_data.take(valid_data.__len__() - 1)
    return valid_data