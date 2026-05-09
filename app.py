from __future__ import annotations

import pandas as pd
import streamlit as st

from predictor_core import SO4Predictor, build_download_bundle, translations


st.set_page_config(page_title="SO4 Predictor", page_icon="🧪", layout="wide")


@st.cache_resource
def load_predictor() -> SO4Predictor:
    return SO4Predictor()


def parse_optional_float(text: str):
    text = str(text).strip()
    if not text:
        return None
    return float(text)


predictor = load_predictor()
lang_label = st.sidebar.radio("Language / 语言", options=["中文", "English"], index=0)
lang = "zh" if lang_label == "中文" else "en"
t = translations(lang)

UI = {
    "zh": {
        "title": "SO4 radical 反应速率常数单分子预测器",
        "caption": "输入一个 SMILES 以及可选实验条件，返回预测 logK、kSO4、可信区间、适用域与使用建议。",
        "inputs": "输入条件",
        "smiles": "SMILES",
        "ph": "pH（可选）",
        "temp": "温度 / °C（可选）",
        "ratio": "pH/T(K)（可选）",
        "neighbors": "最近邻样本数",
        "run": "开始预测",
        "advice": "使用建议",
        "hist": "历史误差参考",
        "near": "最相近训练样本",
        "base": "基学习器预测",
        "json": "原始 JSON 输出",
        "download": "一键下载全部结果（ZIP）",
        "confidence": "预测可信度",
        "fail": "预测失败：",
        "idle": "在左侧输入 SMILES 和可选条件，然后点击“开始预测”。",
        "risk": "风险标签",
        "threshold": "经验适用域阈值",
        "in_domain": "是否落在经验适用域内",
    },
    "en": {
        "title": "SO4 radical Single-Molecule Predictor",
        "caption": "Enter a SMILES string and optional experimental conditions to obtain predicted logK, kSO4, confidence interval, applicability domain, and usage guidance.",
        "inputs": "Inputs",
        "smiles": "SMILES",
        "ph": "pH (optional)",
        "temp": "Temperature / °C (optional)",
        "ratio": "pH/T(K) (optional)",
        "neighbors": "Nearest neighbors",
        "run": "Run prediction",
        "advice": "Usage guidance",
        "hist": "Historical error reference",
        "near": "Nearest reference neighbors",
        "base": "Base-model predictions",
        "json": "Raw JSON output",
        "download": "Download all outputs (ZIP)",
        "confidence": "Confidence",
        "fail": "Prediction failed: ",
        "idle": "Enter a SMILES string and optional conditions in the sidebar, then click Run prediction.",
        "risk": "Risk flags",
        "threshold": "Empirical AD threshold",
        "in_domain": "Empirical in-domain",
    },
}[lang]

st.title(UI["title"])
st.caption(UI["caption"])

with st.sidebar:
    st.subheader(UI["inputs"])
    smiles = st.text_area(UI["smiles"], value="CCO", height=120)
    p_h_text = st.text_input(UI["ph"], value="")
    temp_c_text = st.text_input(UI["temp"], value="")
    p_h_over_tk_text = st.text_input(UI["ratio"], value="")
    top_k = st.slider(UI["neighbors"], min_value=3, max_value=10, value=5)
    run = st.button(UI["run"], type="primary")

st.markdown("---")

if run:
    try:
        report = predictor.predict(
            smiles=smiles.strip(),
            p_h=parse_optional_float(p_h_text),
            temp_c=parse_optional_float(temp_c_text),
            p_h_over_tk=parse_optional_float(p_h_over_tk_text),
            top_k_neighbors=top_k,
        )
        pred = report["prediction"]
        app = report["applicability"]
        err = report["expected_error_reference"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(t["pred_logk"], f"{pred['pred_logK']:.4f}")
        c2.metric(t["pred_k"], f"{pred['pred_kSO4']:.3e}")
        c3.metric(t["nn_similarity"], f"{app['nn_similarity']:.3f}")
        c4.metric(t["domain_level"], app["domain_level"])
        c5.metric(UI["confidence"], t.get(app["confidence_level"], app["confidence_level"]))

        c6, c7, c8 = st.columns(3)
        c6.metric("CI lower", f"{pred['ci_lower']:.4f}")
        c7.metric("CI upper", f"{pred['ci_upper']:.4f}")
        c8.metric(t["sigma"], f"{pred['ensemble_disagreement_sigma']:.4f}")

        if app["confidence_level"] == "high-confidence":
            st.success(t.get(app["confidence_level"], app["confidence_level"]))
        elif app["confidence_level"] == "medium-confidence":
            st.warning(t.get(app["confidence_level"], app["confidence_level"]))
        else:
            st.error(t.get(app["confidence_level"], app["confidence_level"]))

        st.subheader(UI["advice"])
        st.write(t.get(report["recommended_use"], report["recommended_use"]))
        st.write(f"{UI['risk']}：{', '.join(report['risk_flags']) if report['risk_flags'] else t['none']}")
        st.write(f"{UI['threshold']}：{app['empirical_ad_threshold']:.3f}；{UI['in_domain']}：{app['empirical_in_domain']}")

        st.subheader(UI["hist"])
        e1, e2 = st.columns(2)
        e1.metric(t["random_err"], f"{err['random_mean_abs_error']:.3f}")
        e2.metric(t["scaffold_err"], f"{err['scaffold_mean_abs_error']:.3f}")

        st.subheader(UI["near"])
        st.dataframe(pd.DataFrame(report["neighbors"]), use_container_width=True)

        st.subheader(UI["base"])
        st.dataframe(pd.DataFrame([report["base_model_predictions"]]), use_container_width=True)

        zip_bytes = build_download_bundle(report, lang=lang)
        st.download_button(
            label=UI["download"],
            data=zip_bytes,
            file_name="so4_prediction_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

        with st.expander(UI["json"]):
            st.json(report)

    except Exception as exc:
        st.error(UI["fail"] + str(exc))
else:
    st.info(UI["idle"])

