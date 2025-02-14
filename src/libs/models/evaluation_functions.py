from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score
import tensorflow as tf

def evaluate_model(model, X_test, y_test, batch_size=128):
    with tf.device('/CPU:0'):
        y_pred = np.argmax(model.predict(X_test, batch_size=batch_size), axis=-1)

        print("Evaluating Model")

        print(y_test.shape, y_pred.shape)

        cm = confusion_matrix(y_test, y_pred)

        # Compute additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=1.0)
        f1 = f1_score(y_test, y_pred, pos_label=1.0)

        print("Confusion Matrix:")
 

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nAccuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred  
        }
    

def predict_fraud(model, X_test, y_test, batch_size=32):
    with tf.device('/CPU:0'):
        y_pred = np.argmax(model.predict(X_test, batch_size=batch_size), axis=-1)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        return y_pred
    