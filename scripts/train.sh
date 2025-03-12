# Paddle Object Detection
python main.py -c /home/titan/Workspace/Paddle/PaddleX/paddlex/configs/modules/object_detection/PP-YOLOE_plus-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=/home/titan/Workspace/Paddle/PaddleX/datasets/bottles \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe

python main.py -c /home/titan/Workspace/Paddle/PaddleX/paddlex/configs/modules/object_detection/PP-YOLOE_plus-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/home/titan/Workspace/Paddle/PaddleX/datasets/bottles


python main.py -c /home/titan/Workspace/Paddle/PaddleX/paddlex/configs/modules/object_detection/PP-YOLOE_plus-L.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="/home/titan/Workspace/Paddle/PaddleX/datasets/test/233.jpg"

# PaddleOCR Text Detection
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./datasets/train_data/det

python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./datasets/train_data/det

python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./datasets/train_data/det \
    -o Evaluate.weight_path=./models/detection/best_accuracy/best_accuracy.pdparams

python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./models/detection/best_accuracy/inference" \
    -o Predict.input="datasets/test/233.jpg"

# PaddleOCR Text Recognition
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./datasets/train_data/rec

python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./datasets/train_data/rec

python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./datasets/train_data/rec \
    -o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams

python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="datasets/train_data/rec/val/854_crop_3.jpg"


    