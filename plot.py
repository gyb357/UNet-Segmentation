import pandas as pd
import matplotlib.pyplot as plt


file_path = 'csv/train_logg.csv'
df = pd.read_csv(file_path)


plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(df['Epoch'], df['Train_loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Val_loss'], label='Validation Loss')
plt.legend()


plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(df['Epoch'], df['Train_metrics'], label='Train Accuracy')
plt.plot(df['Epoch'], df['Val_metrics'], label='Validation Accuracy')
plt.legend()


plt.tight_layout()
plt.show()

