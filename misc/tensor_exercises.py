import torch
import numpy as np

def check(task_name, tensor, expected_shape, expected_value=None):
    print(f"\n--- {task_name} ---")
    if tensor is None:
        print("❌ Not implemented yet")
        return
    
    # Check shape
    if hasattr(tensor, 'shape'):
        print(f"Your shape:     {tuple(tensor.shape)}")
        print(f"Expected shape: {expected_shape}")
        if tuple(tensor.shape) == expected_shape:
            print("✅ Shape correct!")
        else:
            print("❌ Shape mismatch")
    elif isinstance(tensor, (int, float)):
        print(f"Your types:     {type(tensor)}")
        print(f"Expected type:  python scalar ({type(expected_value)})")
        # For Scalars
        if expected_shape is None: # Scalar check
            print("✅ It is a scalar!")
        else:
             print("❌ Expected a tensor but got scalar")

    # Check value if provided
    if expected_value is not None:
        if isinstance(tensor, torch.Tensor):
            vals_match = torch.allclose(tensor, torch.tensor(expected_value).float()) if tensor.is_floating_point() else torch.equal(tensor, torch.tensor(expected_value))
        else:
            vals_match = (tensor == expected_value)
        
        if vals_match:
            print(f"✅ Value correct: {tensor}")
        else:
            print(f"❌ Value mismatch. Expected:\n{expected_value}\nGot:\n{tensor}")


print("🔥 PYTORCH DIMENSION GYM 🔥\n")

# ==========================================
# LEVEL 1: SLICING & INDEXING (The classic confusion)
# ==========================================
# Представь, что это батч текстов.
# (Batch_Size=2, Seq_Len=3, Vocab_Size=4)
# Батч из 2 предложений, в каждом 3 слова, каждое слово - вектор из 4 чисел (логиты).
logits = torch.tensor([
    [[0.1, 0.2, 0.3, 0.9], [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.0, 0.0]], # Предложение 1
    [[0.8, 0.1, 0.0, 0.1], [0.2, 0.2, 0.2, 0.4], [0.0, 0.0, 1.0, 0.0]]  # Предложение 2
])
print(f"Original Tensor 'logits':\n{logits}")
print(f"Shape: {logits.shape}  (Batch, Seq, Vocab)\n")


# ЗАДАНИЕ 1.1: Получить последнее слово (вектор) для каждого предложения.
# НО! Мы хотим, чтобы размерность времени ИСЧЕЗЛА (сплющилась).
# Ожидаемая форма: (2, 4) -> (Batch, Vocab)
# Подсказка: используй обычный индекс -1
# TODO: напиши код
res_1_1 = None 
# res_1_1 = logits[...] 

check("1.1. Last token (flattened)", res_1_1, (2, 4))


# ЗАДАНИЕ 1.2: Получить последнее слово для каждого предложения.
# НО! Мы хотим СОХРАНИТЬ размерность времени.
# Это нужно для конкатенации (cat) в будущем.
# Ожидаемая форма: (2, 1, 4) -> (Batch, Seq=1, Vocab)
# Подсказка: используй срез (slice) -1:
# TODO: напиши код
res_1_2 = None
# res_1_2 = logits[...]

check("1.2. Last token (keep dims)", res_1_2, (2, 1, 4))


# ==========================================
# LEVEL 2: SQUEEZE & UNSQUEEZE (Игра на гармошке)
# ==========================================

# ЗАДАНИЕ 2.1: У нас есть тензор res_1_2 формы (2, 1, 4).
# Убери лишнюю размерность посередине (которая равна 1) с помощью squeeze.
# Ожидаемая форма: (2, 4)
# TODO: напиши код
res_2_1 = None
# res_2_1 = ...

check("2.1. Squeeze the middle", res_2_1, (2, 4))


# ЗАДАНИЕ 2.2: У тебя есть вектор одного слова (Vocab_Size=4).
word_vec = torch.tensor([0.1, 0.2, 0.3, 0.4]) # Shape: (4,)
# Ты хочешь подать его в RNN, которая ждет батч (Batch, Seq, Feature).
# Сделай из него "фальшивый" батч из 1 примера длиной 1.
# Ожидаемая форма: (1, 1, 4)
# Подсказка: используй unsqueeze два раза или view/reshape
# TODO: напиши код
res_2_2 = None
# res_2_2 = ...

check("2.2. Unsqueeze to batch", res_2_2, (1, 1, 4))


# ==========================================
# LEVEL 3: .item() (Вытаскиваем число из матрицы)
# ==========================================
loss_tensor = torch.tensor([[2.5]]) # Shape (1, 1)

# ЗАДАНИЕ 3.1: Получи обычное число (float) из этого тензора.
# Ожидаемый тип: float, значение 2.5
# Подсказка: .item()
# TODO: напиши код
res_3_1 = None
# res_3_1 = ...

check("3.1. Extract scalar", res_3_1, None, expected_value=2.5)


# ==========================================
# LEVEL 4: TORCH.CAT (Склейка)
# ==========================================
# У нас есть история (context) и новое предсказанное слово (new).
context = torch.zeros(1, 5, 4) # (Batch=1, Seq=5, Feature=4)
new_word = torch.ones(1, 1, 4)  # (Batch=1, Seq=1, Feature=4)

# ЗАДАНИЕ 4.1: Приклей новое слово в КОНЕЦ последовательности.
# То есть мы увеличиваем длину последовательности (Seq_Len).
# Ожидаемая форма: (1, 6, 4)
# Подсказка: укажи правильный dim=?
# TODO: напиши код
res_4_1 = None
# res_4_1 = torch.cat([...], dim=...)

check("4.1. Concat / Append", res_4_1, (1, 6, 4))
