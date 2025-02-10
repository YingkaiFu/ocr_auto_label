import numpy as np
import onnxruntime
import mmcv
from utils import preprocess_image, process_output, parse_dict


inference_result = []  # 存储推理结果

onnx_path = r"ocr_infer.onnx"
model_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # 加载ONNX模型


def infer_image(model_session, preprocessed_img, dictionary):
    inputs = {model_session.get_inputs()[0].name: preprocessed_img}

    outputs = model_session.run(None, inputs)[0]

    res_data = process_output(outputs, dictionary=dictionary)
    return res_data


def rec_infer_roi(img_path, roi):
    img = mmcv.imread(img_path)
    mean = np.array([127, 127, 127], dtype=np.float32)
    std = np.array([127, 127, 127], dtype=np.float32)
    rec_results = []
    input_shape = [1, 3, 32, 320]
    # 预处理图像，包括裁剪和归一化处理
    preprocessed_img = preprocess_image(img, input_shape, mean, std,roi=roi)
    res_data = infer_image(model_session, preprocessed_img, parse_dict("common"))
    rec_results.append(res_data['label_name'])

    return rec_results

