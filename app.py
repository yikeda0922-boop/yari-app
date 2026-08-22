import streamlit as st
import pickle
import pandas as pd
import os

# ページ基本設定
st.set_page_config(
    page_title="やまびより安全判定アプリ",
    page_icon="⛰️",
    layout="centered"
)

# ヘッダー・タイトル
st.title("⛰️ やまびより安全判定アプリ")
st.caption("今日登れる？槍・白根・塔ノ岳の山頂コンディションをAI判定")
st.write("各山域の気象予報を入力すると、山頂の安全度（Go / Caution / No-Go）を機械学習モデルが予測します。")

# 1. 山の選択
mountain = st.selectbox(
    "判定したい山を選択してください",
    ["槍ヶ岳 (3,180m)", "日光白根山 (2,578m)", "塔ノ岳 (1,491m)"]
)

# 2. 選択された山に応じた設定
mountain_config = {
    "槍ヶ岳 (3,180m)": {
        "model_candidates": ["model_yari.pkl"],
        "lowland_name": "安曇野（穂高）",
        "precip_name": "上高地の降水量の合計",
        "precip_label": "上高地の予想降水量 (mm)",
        "image_candidates": ["yari.JPG", "yari.jpg", "yari.png", "yari.jpeg"],
        "caption": "槍ヶ岳（北アルプス・標高3,180m）"
    },
    "日光白根山 (2,578m)": {
        "model_candidates": ["model_nikko.pkl"],
        "lowland_name": "日光東町",
        "precip_name": "奥日光の降水量の合計",
        "precip_label": "奥日光の予想降水量 (mm)",
        "image_candidates": ["nikko.jpg", "nikko.JPG", "nikko.png", "nikko.jpeg"],
        "caption": "日光白根山（関東以北最高峰・標高2,578m）"
    },
    "塔ノ岳 (1,491m)": {
        # tonodake / tounodake どちらのファイル名でも読み込めるように設定
        "model_candidates": ["model_tonodake.pkl", "model_tonodake.pkl"],
        "lowland_name": "海老名",
        "precip_name": "丹沢湖の降水量の合計",
        "precip_label": "丹沢湖の予想降水量 (mm)",
        "image_candidates": ["tonodake.jpg", "tonodake.JPG", "tonodake.png", "tonodake.jpeg"],
        "caption": "塔ノ岳（丹沢山地・標高1,491m）"
    }
}

config = mountain_config[mountain]

# 画像の表示
found_image = None
for img in config["image_candidates"]:
    if os.path.exists(img):
        found_image = img
        break

if found_image:
    st.image(found_image, caption=config["caption"], use_container_width=True)

# 3. モデルの読み込み
@st.cache_resource
def load_model(candidates):
    for fn in candidates:
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f"モデルファイルが見つかりません: {candidates}")

model = load_model(config["model_candidates"])

# 4. 気象データ入力フォーム
st.subheader(f"📍 {config['lowland_name']}（平地・周辺）の気象予報を入力")

col1, col2 = st.columns(2)
with col1:
    max_temp = st.number_input(f"{config['lowland_name']}の最高気温 (℃)", value=20.0, step=0.5)
    min_temp = st.number_input(f"{config['lowland_name']}の最低気温 (℃)", value=10.0, step=0.5)
with col2:
    max_wind = st.number_input(f"{config['lowland_name']}の最大風速 (m/s)", value=3.0, step=0.5, min_value=0.0)
    precip = st.number_input(config["precip_label"], value=0.0, step=1.0, min_value=0.0)

# 5. 判定実行と結果表示
if st.button("登山安全度を判定する", type="primary", use_container_width=True):
    # モデルが学習した特徴量名に合わせてDataFrameを作成
    input_data = pd.DataFrame([{
        '最高気温(℃)': max_temp,
        '最低気温(℃)': min_temp,
        '最大風速(m/s)': max_wind,
        config['precip_name']: precip
    }])
    
    # 予測
    prediction = model.predict(input_data)[0]
    
    st.divider()
    st.subheader(f"【{mountain}】の判定結果")
    
    if prediction == "Go":
        st.success(f"### 判定: 【 {prediction} 】 (登山好適)")
        st.write("天候条件は良好です。標準的な登山装備を整えて行動してください。")
    elif prediction == "Caution":
        st.warning(f"### 判定: 【 {prediction} 】 (注意・警戒)")
        st.write("低温・雨・強風のいずれかのリスクがあります。悪天候時の撤退判断や防寒・雨具を厳重に準備してください。")
    else:
        st.error(f"### 判定: 【 {prediction} 】 (登山不適・危険)")
        st.write("荒天または極端な低温の恐れがあります。入山の中止や延期を強く推奨します。")
