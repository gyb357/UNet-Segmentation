import pandas as pd
import matplotlib.pyplot as plt


file_path = 'csv/train_logg.csv'
df = pd.read_csv(file_path)

plt.figure()
plt.plot(df['Epoch'], df['Train_loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Val_loss'], label='Validation Loss')
plt.plot(df['Epoch'], df['Train_metrics'], label='Train metrics')
plt.plot(df['Epoch'], df['Val_metrics'], label='Validation metrics')

plt.title('Training and Validation Loss/Metrics over Epochs')

plt.xlabel('Epoch')
plt.ylabel('Value')

plt.legend()
plt.tight_layout()
plt.show()

