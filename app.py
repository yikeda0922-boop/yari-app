import streamlit as st
import pickle
import pandas as pd

# ページの基本設定
st.set_page_config(page_title="槍ヶ岳 登山安全判定アプリ", page_icon="⛰️")

# タイトル
st.title("⛰️ 槍ヶ岳 登山安全判定アプリ")

# 自身で撮影した槍ヶ岳の写真を表示
st.image("yari.jpg", caption="槍ヶ岳（標高3,180m）", use_container_width=True)
st.write("安曇野（穂高）の気象予報を入力すると、槍ヶ岳山頂の安全度（Go / Caution / No-Go）をAIが予測します。")

# 1. 学習済みモデルの読み込み
@st.cache_resource
def load_model():
    with open('model_yari.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 2. ユーザー入力フォーム
st.subheader("📍 穂高（平地）の気象予報を入力")

col1, col2 = st.columns(2)
with col1:
    temp_max = st.number_input("最高気温 (℃)", value=28.0, step=0.5)
    temp_min = st.number_input("最低気温 (℃)", value=18.0, step=0.5)
with col2:
    wind_max = st.number_input("最大風速 (m/s)", value=4.0, step=0.5)
    precip = st.number_input("降水量の合計 (mm)", value=0.0, step=0.5)

# 3. 判定ボタンと予測
if st.button("登山安全度を判定する", type="primary"):
    # 降水量列名を自動取得
    precip_col_name = '上高地の降水量の合計' if '上高地の降水量の合計' in model.feature_names_in_ else '降水量'

    input_data = pd.DataFrame([[temp_max, temp_min, wind_max, precip]],
                              columns=['最高気温', '最低気温', '最大風速', precip_col_name])

    # 予測実行
    pred = model.predict(input_data)[0]

    # 結果の表示
    st.markdown("---")
    st.subheader("🎯 判定結果")

    if pred == 'Go':
        st.success("🟢 **Go（登山適日）**\n\n岩場・稜線ともに安全に通行できる良好な気象条件です。")
    elif pred == 'Caution':
        st.warning("🟡 **Caution（注意・装備徹底）**\n\n雨具必携。稜線の強風や冷え込みに十分警戒してください。")
    else:
        st.error("🔴 **No-Go（登山回避推奨）**\n\n暴風・大雨・氷点下などの重大事故リスクがあります。行動の中止や停滞を推奨します。")
