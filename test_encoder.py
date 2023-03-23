from hoogberta.syllable_encoder import HoogBERTaSyllableEncoder
import os 
model_home = os.environ.get('HoogBERTaSyllable')

model = HoogBERTaSyllableEncoder(base_path=model_home)
model = model.cuda()
batch, output = model.extract_features("สวัสดีครับ123")
print(output)

print(model.string(batch))

batch, output = model.extract_features_batch(["สวัสดีครับ","สวัสดีค่ะ"])
print(output)

#x = model.roberta.fill_mask('สวัสดีครับ123 <mask>', topk=3)
#print(x)