import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# -------------------------- 各Agent类定义 --------------------------
class DataLoaderAgent:
    def generate_test_data(self):
        np.random.seed(0)
        img = np.zeros((256, 256), dtype=np.uint8)
        for _ in range(15):
            cx, cy = np.random.randint(30, 226, 2)
            r = np.random.randint(8, 15)
            cv2.circle(img, (cx, cy), r, 255, -1)
        return img

    def load_uploaded_data(self, uploaded_file):
        image = Image.open(uploaded_file).convert("L")
        return np.array(image)


class PreprocessAgent:
    def process(self, raw_img):
        blurred = cv2.GaussianBlur(raw_img, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


class FeatureExtractionAgent:
    def extract(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        features = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            features.append({
                "ID": idx + 1,
                "面积": round(area, 2),
                "周长": round(perimeter, 2),
                "圆度": round(circularity, 2),
                "中心": (cx, cy)
            })
            cv2.putText(label_img, str(idx+1), (cx-5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return features, label_img


class AnalysisAgent:
    def analyze(self, features):
        wbc_list = []
        rbc_list = []
        for cell in features:
            if cell["面积"] > 200:
                wbc_list.append(cell)
            else:
                rbc_list.append(cell)
        analysis_res = {
            "细胞总数": len(features),
            "白细胞数": len(wbc_list),
            "红细胞数": len(rbc_list),
            "平均面积": round(np.mean([c["面积"] for c in features]), 2) if features else 0,
            "平均圆度": round(np.mean([c["圆度"] for c in features]), 2) if features else 0
        }
        return analysis_res, wbc_list, rbc_list


class VisualReportAgent:
    def generate_figure(self, raw_img, binary_img, label_img, features, analysis_res, wbc_list, rbc_list):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(raw_img, cmap="gray")
        axes[0].set_title("原始图像")
        axes[0].axis("off")

        axes[1].imshow(binary_img, cmap="gray")
        axes[1].set_title("二值化图像")
        axes[1].axis("off")

        axes[2].imshow(label_img)
        axes[2].set_title("标记图像")
        axes[2].axis("off")

        plt.tight_layout()
        return fig

# -------------------------- 界面主体 --------------------------
st.title("多Agent自动化光学图像分析系统")
st.markdown("### 功能说明：支持自动生成测试图像 / 上传本地光学图像，完成细胞识别、统计与可视化分析")

# 初始化session_state
if "raw_image" not in st.session_state:
    st.session_state.raw_image = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

data_source = st.radio("请选择数据来源", ["自动生成测试数据", "上传自定义图像"])

loader = DataLoaderAgent()
if data_source == "上传自定义图像":
    uploaded_file = st.file_uploader("上传图片（PNG/JPG/TIF）", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file:
        st.session_state.raw_image = loader.load_uploaded_data(uploaded_file)
        st.image(st.session_state.raw_image, caption="已上传图像", width=700)
else:
    if st.button("点击生成测试图像"):
        st.session_state.raw_image = loader.generate_test_data()
        st.image(st.session_state.raw_image, caption="自动生成光学细胞图像", width=700)

# 开始分析
if st.session_state.raw_image is not None and st.button("开始智能分析"):
    st.session_state.analysis_done = True
    with st.spinner("多Agent协同分析中，请稍候..."):
        preprocessor = PreprocessAgent()
        binary_img = preprocessor.process(st.session_state.raw_image)

        extractor = FeatureExtractionAgent()
        features, label_img = extractor.extract(binary_img)

        analyzer = AnalysisAgent()
        analysis_res, wbc_list, rbc_list = analyzer.analyze(features)

        reporter = VisualReportAgent()
        fig = reporter.generate_figure(st.session_state.raw_image, binary_img, label_img, features, analysis_res, wbc_list, rbc_list)

    st.success("分析完成！")
    st.subheader("整体统计结果")
    st.write(analysis_res)

    st.subheader("综合可视化图表")
    st.pyplot(fig)

    # 下载报告
    report_content = "===== 多Agent光学图像分析报告 =====\n\n"
    for k, v in analysis_res.items():
        report_content += f"{k}：{v}\n"
    report_content += "\n===== 单细胞明细特征 =====\n"
    for cell in features:
        report_content += f"ID:{cell['ID']}  面积:{cell['面积']}  周长:{cell['周长']}  圆度:{cell['圆度']}\n"

    st.download_button(
        label="下载完整分析报告(TXT)",
        data=report_content,
        file_name="分析报告.txt",
        mime="text/plain"
    )
