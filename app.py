import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters, morphology
from skimage.draw import disk
from PIL import Image

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 多Agent 模块 =====================
class DataLoaderAgent:
    def generate_test_data(self):
        img = np.ones((800, 800), dtype=np.uint8) * 20
        np.random.seed(42)
        # 生成白细胞
        for _ in range(15):
            x, y = np.random.randint(50, 750, 2)
            rr, cc = disk((x, y), 20)
            img[rr, cc] = 180
        # 生成红细胞
        for _ in range(30):
            x, y = np.random.randint(50, 750, 2)
            rr, cc = disk((x, y), 10)
            img[rr, cc] = 220
        # 添加模拟噪声
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        return img

    def load_uploaded_data(self, uploaded_file):
        img = Image.open(uploaded_file).convert('L')
        return np.array(img)

class PreprocessAgent:
    def process(self, raw_image):
        denoised = cv2.GaussianBlur(raw_image, (5, 5), 0)
        binary = filters.threshold_otsu(denoised)
        binary_img = denoised > binary
        cleaned = morphology.remove_small_objects(binary_img, min_size=50)
        return cleaned.astype(np.uint8) * 255

class FeatureExtractionAgent:
    def extract(self, binary_image):
        labels = measure.label(binary_image)
        props = measure.regionprops(labels)
        features = []
        for prop in props:
            circ = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            feature = {
                "ID": prop.label,
                "面积": round(prop.area, 2),
                "周长": round(prop.perimeter, 2),
                "圆度": round(circ, 3),
                "偏心距": round(prop.eccentricity, 3),
                "外接框": prop.bbox
            }
            features.append(feature)
        return features, labels

class AnalysisAgent:
    def analyze(self, features):
        wbc = [f for f in features if f["面积"] > 300]
        rbc = [f for f in features if f["面积"] <= 300]
        result = {
            "总细胞数": len(features),
            "白细胞数": len(wbc),
            "红细胞数": len(rbc),
            "白细胞平均面积": round(np.mean([f["面积"] for f in wbc]), 2) if wbc else 0,
            "红细胞平均面积": round(np.mean([f["面积"] for f in rbc]), 2) if rbc else 0,
            "白细胞平均圆度": round(np.mean([f["圆度"] for f in wbc]), 3) if wbc else 0,
            "红细胞平均圆度": round(np.mean([f["圆度"] for f in rbc]), 3) if rbc else 0
        }
        return result, wbc, rbc

class VisualReportAgent:
    def draw_marked_image(self, raw_gray, wbc, rbc):
        img_color = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
        for cell in wbc:
            y1, x1, y2, x2 = cell["外接框"]
            cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for cell in rbc:
            y1, x1, y2, x2 = cell["外接框"]
            cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img_color

    def generate_figure(self, raw_img, binary_img, label_img, features, analysis_res, wbc, rbc):
        marked_img = self.draw_marked_image(raw_img, wbc, rbc)
        count_data = [analysis_res["白细胞数"], analysis_res["红细胞数"]]
        count_label = ["白细胞", "红细胞"]
        area_all = [f["面积"] for f in features]
        circ_all = [f["圆度"] for f in features]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes[0,0].imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title("原始图像 + 细胞标注（蓝=白细胞 红=红细胞）", fontsize=10)
        axes[0,0].axis("off")

        axes[0,1].imshow(binary_img, cmap="gray")
        axes[0,1].set_title("预处理后二值图像", fontsize=10)
        axes[0,1].axis("off")

        axes[0,2].imshow(label_img, cmap="nipy_spectral")
        axes[0,2].set_title("细胞连通域标记", fontsize=10)
        axes[0,2].axis("off")

        bars = axes[1,0].bar(count_label, count_data, color=["royalblue", "crimson"])
        axes[1,0].set_title("细胞分类数量统计", fontsize=10)
        for bar in bars:
            h = bar.get_height()
            axes[1,0].text(bar.get_x()+bar.get_width()/2, h, f"{h}", ha="center", va="bottom")

        axes[1,1].hist(area_all, bins=15, color="orange", alpha=0.7)
        axes[1,1].set_title("细胞面积分布直方图")
        axes[1,1].set_xlabel("面积(像素)")
        axes[1,1].set_ylabel("数量")

        axes[1,2].hist(circ_all, bins=15, color="green", alpha=0.7)
        axes[1,2].set_title("细胞圆度分布直方图")
        axes[1,2].set_xlabel("圆度")
        axes[1,2].set_ylabel("数量")

        plt.tight_layout()
        return fig

# ===================== Web 界面 =====================
st.title(" 多Agent自动化光学图像分析系统")
st.markdown("### 功能说明：支持自动生成测试图像 / 上传本地光学图像，完成细胞识别、统计与可视化分析")

data_source = st.radio("请选择数据来源", ["自动生成测试数据", "上传自定义图像"])
raw_image = None

loader = DataLoaderAgent()
if data_source == "上传自定义图像":
    uploaded_file = st.file_uploader("上传图片（PNG/JPG/TIF）", type=["png","jpg","jpeg","tif"])
    if uploaded_file:
        raw_image = loader.load_uploaded_data(uploaded_file)
        st.image(raw_image, caption="已上传图像", use_column_width=True)
else:
    if st.button("点击生成测试图像"):
        raw_image = loader.generate_test_data()
        st.image(raw_image, caption="自动生成光学细胞图像", use_column_width=True)

# 开始分析
if raw_image is not None and st.button("开始智能分析"):
    with st.spinner("多Agent协同分析中，请稍候..."):
        preprocessor = PreprocessAgent()
        binary_img = preprocessor.process(raw_image)

        extractor = FeatureExtractionAgent()
        features, label_img = extractor.extract(binary_img)

        analyzer = AnalysisAgent()
        analysis_res, wbc_list, rbc_list = analyzer.analyze(features)

        reporter = VisualReportAgent()
        fig = reporter.generate_figure(raw_image, binary_img, label_img, features, analysis_res, wbc_list, rbc_list)

    st.success(" 分析完成！")
    st.subheader(" 整体统计结果")
    st.write(analysis_res)

    st.subheader(" 综合可视化图表")
    st.pyplot(fig)

    # 生成可下载报告
    report_content = "===== 多Agent光学图像分析报告 =====\n\n"
    for k, v in analysis_res.items():
        report_content += f"{k}：{v}\n"
    report_content += "\n===== 单细胞明细特征 =====\n"
    for cell in features:
        report_content += f"ID:{cell['ID']}  面积:{cell['面积']}  周长:{cell['周长']}  圆度:{cell['圆度']}\n"

    st.download_button(
        label=" 下载完整分析报告(TXT)",
        data=report_content,
        file_name="分析报告.txt",
        mime="text/plain"
    )