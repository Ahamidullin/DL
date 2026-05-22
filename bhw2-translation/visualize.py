"""
Визуализация результатов обучения baseline модели.
Запускать после обучения для генерации графиков в отчёт.
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================
# Данные из логов обучения (baseline, 10 эпох)
# =============================================
epochs = list(range(1, 11))

train_loss = [4.3571, 3.0297, 2.5719, 2.3494, 2.2665, 2.2326, 2.2392, 2.2669, 2.3101, 2.3574]
val_loss   = [6.5666, 5.9585, 5.7290, 5.5746, 5.4847, 5.3278, 5.3334, 5.1185, 5.0728, 4.9343]
bleu       = [12.96,  21.59,  23.42,  23.80,  23.48,  24.02,  24.24,  22.98,  23.36,  22.62]
ppl        = [710.98, 387.02, 307.66, 263.64, 240.99, 205.99, 207.14, 167.08, 159.62, 138.97]
tf_ratio   = [1.00,   0.94,   0.89,   0.83,   0.78,   0.72,   0.67,   0.61,   0.56,   0.50]

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Baseline: BiGRU + Bahdanau Attention (10 эпох)', fontsize=16, fontweight='bold')

# --- 1. Train & Val Loss ---
ax = axes[0, 0]
ax.plot(epochs, train_loss, 'o-', color='#2196F3', label='Train Loss', linewidth=2, markersize=6)
ax.plot(epochs, val_loss, 's-', color='#F44336', label='Val Loss', linewidth=2, markersize=6)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (CrossEntropy)')
ax.set_title('Train / Val Loss')
ax.legend()
ax.set_xticks(epochs)

# --- 2. BLEU ---
ax = axes[0, 1]
ax.plot(epochs, bleu, 'D-', color='#4CAF50', linewidth=2, markersize=6)
best_epoch = np.argmax(bleu) + 1
best_bleu = max(bleu)
ax.axhline(y=best_bleu, color='#4CAF50', linestyle='--', alpha=0.5)
ax.annotate(f'Best: {best_bleu:.2f} (ep.{best_epoch})',
            xy=(best_epoch, best_bleu), xytext=(best_epoch + 1.5, best_bleu + 1),
            arrowprops=dict(arrowstyle='->', color='#333'),
            fontsize=11, fontweight='bold', color='#333')
ax.axhline(y=24, color='orange', linestyle=':', alpha=0.7, label='Порог соревнования (24)')
ax.set_xlabel('Epoch')
ax.set_ylabel('BLEU-4')
ax.set_title('Validation BLEU')
ax.legend()
ax.set_xticks(epochs)

# --- 3. Perplexity ---
ax = axes[1, 0]
ax.plot(epochs, ppl, '^-', color='#9C27B0', linewidth=2, markersize=6)
ax.set_xlabel('Epoch')
ax.set_ylabel('Perplexity')
ax.set_title('Validation Perplexity')
ax.set_xticks(epochs)

# --- 4. Teacher Forcing Ratio ---
ax = axes[1, 1]
ax.plot(epochs, tf_ratio, 'v-', color='#FF9800', linewidth=2, markersize=6)
ax.fill_between(epochs, tf_ratio, alpha=0.2, color='#FF9800')
ax.set_xlabel('Epoch')
ax.set_ylabel('TF Ratio')
ax.set_title('Teacher Forcing Schedule')
ax.set_xticks(epochs)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('baseline_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("Сохранено: baseline_training.png")


# =============================================
# Таблица с примерами переводов
# =============================================
print("\n" + "=" * 80)
print("Примеры переводов (best model, epoch 7, BLEU 24.24)")
print("=" * 80)

examples = [
    {
        "src": "als ich 11 jahre alt war , wurde ich eines morgens von den klängen heller freude geweckt .",
        "hyp": "when i was 11 years old , i grew one morning by the morning .",
        "ref": "when i was 11 , i remember waking up one morning to the sound of joy in my house .",
    },
    {
        "src": 'er rief : " die taliban sind weg ! "',
        "hyp": 'he called " the taliban are gone . "',
        "ref": '" the taliban are gone ! " my father shouted .',
    },
    {
        "src": "ich wusste nicht , was das bedeutete , aber es machte meinen vater offensichtlich sehr , sehr glücklich .",
        "hyp": "i didn 't know what that meant , but it made my father very very , very happy .",
        "ref": "i didn 't know what it meant , but i could see that my father was very , very happy .",
    },
]

for i, ex in enumerate(examples, 1):
    print(f"\n--- Пример {i} ---")
    print(f"  SRC: {ex['src']}")
    print(f"  HYP: {ex['hyp']}")
    print(f"  REF: {ex['ref']}")
