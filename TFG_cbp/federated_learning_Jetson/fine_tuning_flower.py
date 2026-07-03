# -------------------------------------------------------------------------------------------------------------
# File Name                : fine_tuning_flower.py
# Author                   : Clara Benejam Pons
# Description              : Fine-tuning of the global federated model on each client's local data using Flower.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------

def finetune_global_model(global_model, client_data, finetune_epochs=5):
    """
    Fine-tune the global model on the client's local data.
    """
    model = tf.keras.models.clone_model(global_model)
    model.set_weights(global_model.get_weights())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    y_train_cat = tf.keras.utils.to_categorical(
        client_data["y_train"].astype(int),
        num_classes=NUM_CLASSES
    )

    y_val_cat = tf.keras.utils.to_categorical(
        client_data["y_val"].astype(int),
        num_classes=NUM_CLASSES
    )

    model.fit(
        client_data["X_train"],
        y_train_cat,
        validation_data=(client_data["X_val"], y_val_cat),
        epochs=finetune_epochs,
        batch_size=32,
        verbose=1
    )

    return model

