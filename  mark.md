模型能力对比-速度测试脚本：
python tools/visualize_ebike_model_compare.py --sample 50 --out data/model_compare_vis
模型输出类别脚本：
python -c 'from ultralytics import YOLO; print(YOLO("models/baseline_best.pt").names); print(YOLO("models/bifpn_best.pt").names); print(YOLO("[yolov8n.pt](http://yolov8n.pt/)").names)'