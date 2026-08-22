import streamlit as st
import pickle
import numpy as np
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
        "precip_label": "上高地の予想降水量 (mm)",
        "image_candidates": ["yari.JPG", "yari.jpg", "yari.png", "yari.jpeg"],
        "caption": "槍ヶ岳（北アルプス・標高3,180m）"
    },
    "日光白根山 (2,578m)": {
        "model_candidates": ["model_nikko.pkl"],
        "lowland_name": "日光東町",
        "precip_label": "奥日光の予想降水量 (mm)",
        "image_candidates": ["nikko.jpg", "nikko.JPG", "nikko.png", "nikko.jpeg"],
        "caption": "日光白根山（関東以北最高峰・標高2,578m）"
    },
    "塔ノ岳 (1,491m)": {
        "model_candidates": ["model_tonodake.pkl", "model_tounodake.pkl"],
        "lowland_name": "海老名",
        "precip_label": "丹沢湖の予想降水量 (mm)",
        "image_candidates": ["tonodake.jpg", "tonodake.JPG", "tonodake.png", "tonodake.jpeg"],
        "caption": "塔ノ岳（丹沢山地・標高1,491m）"
    }
}

config = mountain_config[mountain]

# 画像ファイルの探索と表示
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
st.subheader(f"📍 気象予報データの入力（基準観測地: {config['lowland_name']}）")

# わかりやすい補足説明ボックスを追加
st.info(
    f"💡 **入力の目安**\n"
    f"山頂には気象庁の観測所がないため、麓・周辺地域（**{config['lowland_name']}**など）の天気予報値を入力して山の安全度を機械学習モデルが推測・判定します。\n"
)

coll, col2 = st.columns(2)
with coll:
    max_temp = st.slider(
        f"{config['lowland_name']}の最高気温 (℃)", 
        min_value=-20.0, max_value=40.0, value=20.0, step=0.5,
        help="予想される日中の最高気温です。"
    )
    min_temp = st.slider(
        f"{config['lowland_name']}の最低気温 (℃)", 
        min_value=-30.0, max_value=30.0, value=10.0, step=0.5,
        help="予想される朝晩の最低気温です。"
    )
with col2:
    max_wind = st.slider(
        f"{config['lowland_name']}の最大風速 (m/s)", 
        min_value=0.0, max_value=30.0, value=3.0, step=0.5,
        help="平地周辺での予想最大風速です。"
    )
    precip = st.slider(
        config["precip_label"], 
        min_value=0.0, max_value=100.0, value=0.0, step=0.5,
        help="1日の予想総降水量です。"
    )
# 5. 判定実行と結果表示
if st.button("🏔️ 登山安全度を判定する", type="primary", use_container_width=True):
    # 数値配列として渡す
    input_features = np.array([[max_temp, min_temp, max_wind, precip]])
    
    # 予測実行
    prediction = model.predict(input_features)[0]
    
    st.write("---")
    st.subheader("📋 判定結果とアドバイス")
    
    # 判定結果ごとの表示分岐
    if prediction in ['Go', 2, '2']:
        st.success("### ✅ 判定：Go（登山適正・安全圏）")
        st.markdown(
            """
            * **状況の目安**：周辺の風速・降水・気温ともに安定しており、標準的な登山計画で安全に行動しやすいコンディションです。
            * **行動の注意点**：
              * 良好な予報でも山頂周辺の天候急変や局所的な雷雨には留意してください。
              * 朝晩の冷え込みに備え、防寒具や水分・行動食は十分に携帯しましょう。
            """
        )
        
    elif prediction in ['Caution', 1, '1']:
        st.warning("### ⚠️ 判定：Caution（注意・要装備強化）")
        st.markdown(
            """
            * **状況の目安**：小雨・強風、または冷え込み（低温）の可能性があります。
            * **行動の注意点**：
              * **装備の点検**：レインウェア（上下）、防寒着、ヘッドライト、予備バッテリーを必ず携行してください。
              * **計画の見直し**：稜線上では突風や体感温度の低下に注意し、状況が悪化した場合は無理せず途中撤退やエスケープルートを検討してください。
            """
        )
        
    else:  # No-Go
        st.error("### ⛔ 判定：No-Go（登山中止・延期推奨）")
        st.markdown(
            """
            * **状況の目安**：大雨、稜線での暴風、または著しい低温・凍結などの荒天リスクが高い状態です。
            * **行動の注意点**：
              * **遭難・低体温症・転落のリスクが非常に高いため、登山の延期または中止を強く推奨します。**
              * 山小屋や登山口までの移動自体に土砂崩れや公共交通機関の運休リスクがないかも確認してください。
            """
        )
