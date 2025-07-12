import streamlit as st
from ultralytics import YOLO
import cv2
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
import torch
import cv2
import sys
import numpy as np
from PIL import Image
import pandas as pd
import time

sys.path.append("../src/")
st.set_page_config(layout="wide", page_title="License Plate Image Detection")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(r"models/license_plate_detection_models/best_120_epoch_int8_openvino_model_640", task="detect")
lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
lprnet.to(device)
lprnet.load_state_dict(
    torch.load(
        "streamlit/weights/lprnet_Iter_043200_model.ckpt",
        map_location=lambda storage, loc: storage,
    )["net_state_dict"]
)
lprnet.eval()
# 4 sec -> 77 ms
STN = STNet()
STN.to(device)
STN.load_state_dict(
    torch.load(
        "streamlit/weights/stn_Iter_043200_model.ckpt",
        map_location=lambda storage, loc: storage,
    )["net_state_dict"]
)
STN.eval()
print("Successfully built the network!")


def decode(preds, CHARS):
    """
    This function decodes the labels from the predictions tensor
    """
    pred_labels, labels = [], []
    n = preds.shape[0]
    
    for i in range(n):
        pred = preds[i, :, :]
        m = pred.shape[1]
        pred_label = []

        # using the maximum probability character as the predicted character

        for j in range(m):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        non_repeated = [pred_label[0]]
        prev = pred_label[0]
        # Dropping repeated characters
        for c in pred_label[1:]:
            if (prev == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    prev = c
                continue
            non_repeated.append(c)
            prev = c
        pred_labels.append(non_repeated)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)

    return labels, np.array(pred_labels)



def process_roi(roi, bbox_width, bbox_height):
    im = cv2.resize(roi, roi.shape[:2][::-1])  # fixing the rounding error

    if (bbox_height) / (bbox_width) > 0.4:
        width, height = im.shape[:2]
        half = width // 2
        curr1 = im[:half, : height - height // 20]
        curr2 = im[half:, height // 20 :]
        # print(curr1.shape, curr2.shape  )
        if curr1.shape[0] < curr2.shape[0]:
            curr1 = im[: half + 1, : height - height // 20]
        stacked = np.hstack((curr1, curr2))
        im = cv2.resize(stacked, (94, 24))
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # cv2.imshow('testing', im)
        # cv2.waitKey(0)
    else:
        im = cv2.resize(im, (94, 24))
    # cv2.imshow('testing', im)
    # cv2.waitKey(0)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125  # Normalization

    data = (
        torch.from_numpy(im).float().unsqueeze(0).to(device)
    )  # torch.Size([1, 3, 24, 94])
    transfer = STN(data)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 37, 18)

    labels, pred_labels = decode(preds, CHARS)
    return labels[0].lstrip("-").rstrip("-")


def image_inference(image, model):
    # Process the image
    sum_lpr_time = 0
    sum_yolo_time = 0
    count = 0
    rois = []
    class_labels = []
    confidences = []
    for result in model(image):
        yolo_time = result.speed.get("inference")
        sum_yolo_time += yolo_time
        bboxes = result.boxes

        # Process each bounding box
        for bbox in bboxes.data:
            x1, y1, x2, y2, conf, cls = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
                float(bbox[4]),
                int(bbox[5]),
            )

            # Extract the region of interest (ROI) using the bounding box coordinates
            roi = image[y1:y2, x1:x2, :]
            print(roi.shape)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            # Example: Pass the ROI to another model (dummy processing here)
            lpr_time = time.perf_counter()
            class_label = process_roi(roi, bbox_width, bbox_height)
            lpr_end_time = time.perf_counter()
            sum_lpr_time += lpr_end_time - lpr_time
            # Display the class result of the other model
            # Calculate the width and height of the text

            text_width, text_height = cv2.getTextSize(
                class_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )[0]
            cv2.rectangle(image, (x1, y1 - text_height * 2), (x2, y1), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"{class_label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rois.append(roi)
            class_labels.append(class_label)
            confidences.append(conf)
            print("Detected License Plate:", class_label)
            count += 1
    # Save the logs
    return (
        image,
        rois,
        class_labels,
        confidences,
        sum_lpr_time / count,
        sum_yolo_time / count,
    )


st.title("📷 License Plate Image Detection")
st.markdown("""
### Single Image License Plate Recognition
Upload an image to detect and recognize license plates using YOLOv8 + LPRNet pipeline.

**Features:**
- Real-time detection and OCR
- Authorization status checking
- Confidence scoring
- Processing time metrics
""")
st.markdown("---")


st.subheader("📤 Upload Image File")

image = st.file_uploader("Upload a file", type=["png", "webp", "jpg", "jpeg"])
if image is not None:
    # Convert the uploaded image to numpy array
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)

    # Pass the image array to the image_inference function
    try:
        output_image, rois, class_labels, confidences, avg_lpr_time, avg_yolo_time = (
            image_inference(img_array, model)
        )
    except:
        st.error(
            "⚠️ No license plates detected. Please try again with a clearer image."
        )
    else:
        image_columns = st.columns(2, gap="large")
        # Display the output image
        with image_columns[0]:
            st.image(
                output_image,
                caption="Processed Image",
                channels="BGR",
            )
        with image_columns[1]:
            st.subheader("✅ Detection Results")
            
            # Performance metrics
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("License Plates Detected", len(class_labels))
            with metrics_cols[1]:
                st.metric("LPRNet Time (ms)", round(avg_lpr_time * 1000, 2))
            with metrics_cols[2]:
                st.metric("YOLO Time (ms)", round(avg_yolo_time, 2))
            
            st.markdown("---")
            
            # Detection results
            for i, (label, roi, confidence) in enumerate(zip(class_labels, rois, confidences)):
                with st.container():
                    result_cols = st.columns([2, 1])
                    with result_cols[0]:
                        st.markdown(f"**License Plate {i+1}:** `{label}`")
                        st.markdown(f"**Confidence:** {round(confidence, 3)}")
                    with result_cols[1]:
                        st.image(cv2.resize(roi, (150, 75)), channels="BGR", caption="ROI")
                    st.markdown("---")




else:
    st.info("📁 Please upload an image file to begin license plate detection.")

