from hoogberta.syllable_encoder import HoogBERTaSyllableEncoder

model = HoogBERTaSyllableEncoder(base_path="../")
output = model.extract_features("สวัสดีครับ")
print(output)

