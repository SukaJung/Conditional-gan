from deeplab import *
from dataset import *
import io

dataset_dir = '/home/suka/dataset/catdog/train/'
datas = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        datas.append(filename)

MODEL = DeepLabModel("./deeplabv3_mnv2_pascal_train_aug.tar.gz")     
for i in datas:
    img = run_segmentation(MODEL,dataset_dir+i)
    if type(img) is int:
        continue
    img = Image.fromarray(img)
    buffer = io.BytesIO()
    img.save(buffer, format = "JPEG")
    open("/home/suka/dataset/preprocessed_catdog/"+i, "wb").write(buffer.getvalue())