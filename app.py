import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import io

# ==========================================
# CẤU HÌNH TRANG STREAMLIT ĐẦU TIÊN
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==========================================
# CÁC HÀM CACHE DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bộ nhớ bytes để đảm bảo tính hashable cho Streamlit Cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu huấn luyện lên
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Tải lên tệp chứa các đặc trưng X_1 đến X_14 và cột mục tiêu 'default'"
    )
    
    st.divider()
    
    # 2. Lựa chọn mô hình thuật toán
    model_choice = st.selectbox(
        "Lựa chọn mô hình thuật toán",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Chọn thuật toán Machine Learning muốn huấn luyện."
    )
    
    st.subheader("🛠️ Tham số mô hình AI")
    
    # Cấu hình tham số động dựa theo mô hình đã chọn
    params = {}
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số cây quyết định (n_estimators)", min_value=10, max_value=200, value=100, step=10, help="Số lượng cây phân tích trong rừng.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, help="Độ sâu giới hạn của mỗi cây quyết định.")
        params['random_state'] = st.number_input("Cố định dữ liệu (random_state)", value=42, step=1, help="Đảm bảo kết quả huấn luyện không thay đổi sau mỗi lần chạy.")
    
    elif model_choice == "Decision Tree":
        params['criterion'] = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách nhánh.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=8, help="Độ sâu giới hạn tối đa của cây.")
        params['random_state'] = st.number_input("Cố định dữ liệu (random_state)", value=42, step=1)
        
    elif model_choice == "Logistic Regression":
        params['C'] = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Giá trị nhỏ hơn làm tăng tính điều hòa (giảm overfitting).")
        params['max_iter'] = st.number_input("Số vòng lặp tối đa (max_iter)", value=100, step=50, help="Số lượng vòng lặp tối đa cho các thuật toán tối ưu hội tụ.")
        params['random_state'] = st.number_input("Cố định dữ liệu (random_state)", value=42, step=1)

    # Thêm cấu hình kỹ thuật xử lý mất cân bằng dữ liệu giống notebook
    st.subheader("⚖️ Cân bằng dữ liệu")
    use_smote = st.checkbox("Áp dụng SMOTE (Khuyên dùng)", value=True, help="Tự động tạo mẫu nhân tạo cho nhóm thiểu số (Gian lận) tránh thiên vị mô hình.")

    st.divider()
    
    # 3. Nút kích hoạt huấn luyện duy nhất
    btn_train = st.button("🔥 Huấn luyện Mô hình", type="primary", use_container_width=True, help="Bấm để chạy toàn bộ quy trình tiền xử lý và huấn luyện.")

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Hệ thống Phát hiện & Quản trị Rủi ro Giao dịch Gian lận")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu tài chính, phát hiện hành vi giao dịch bất thường mang tính gian lận tín dụng thông qua mô hình học máy nâng cao.")

# Kiểm tra trạng thái dữ liệu đầu vào
if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu huấn luyện mẫu (ví dụ: `dataset1.csv`) ở thanh cấu hình bên trái để bắt đầu.")
    st.stop()
else:
    # Đọc dữ liệu thô và lưu vào biến chính
    file_bytes = uploaded_file.getvalue()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("Tệp tải lên không hợp lệ hoặc bị lỗi cấu trúc.")
        st.stop()
        
    st.caption(f"📁 Đang sử dụng tệp dữ liệu: **{uploaded_file.name}**")

st.divider()

# Khởi tạo các biến đặc trưng dựa vào cấu trúc được trích xuất từ tập dữ liệu mẫu
features = [f"X_{i}" for i in range(1, 15)]
target = "default"

# Kiểm tra cấu trúc cột của file đầu vào có trùng khớp với thiết kế mô hình không
if not all(col in df_raw.columns for col in features + [target]):
    st.error(f"Cấu trúc tệp không hợp lệ! File cần chứa đầy đủ các cột đặc trưng từ X_1 đến X_14 và cột kết quả '{target}'.")
    st.stop()

# ==========================================
# KHỐI HUẤN LUYỆN (LƯU TRỮ VÀO SESSION STATE)
# ==========================================
if btn_train:
    with st.spinner("Đang xử lý dữ liệu và huấn luyện mô hình... Xin vui lòng đợi."):
        # 1. Chia tách dữ liệu X, y
        X = df_raw[features]
        y = df_raw[target]
        
        # 2. Phân chia tập huấn luyện và kiểm thử (70% Train, 30% Test giống cấu trúc chuẩn)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=params.get('random_state', 42), stratify=y)
        
        # 3. Tiền xử lý chuẩn hóa dữ liệu với StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Áp dụng kỹ thuật cân bằng dữ liệu SMOTE nếu được bật
        if use_smote:
            smote = SMOTE(random_state=params.get('random_state', 42))
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        else:
            X_train_res, y_train_res = X_train_scaled, y_train
            
        # 5. Khởi tạo mô hình theo lựa chọn của người dùng
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                random_state=params['random_state'],
                n_jobs=-1
            )
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(
                criterion=params['criterion'], 
                max_depth=params['max_depth'], 
                random_state=params['random_state']
            )
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(
                C=params['C'], 
                max_iter=int(params['max_iter']), 
                random_state=params['random_state']
            )
            
        # 6. Huấn luyện mô hình
        model.fit(X_train_res, y_train_res)
        
        # 7. Dự báo và đánh giá kiểm tra trên tập dữ liệu Test
        y_pred = model.predict(X_test_scaled)
        
        # 8. Lưu kết quả vào st.session_state để chia sẻ giữa các tab không bị mất dữ liệu
        st.session_state['trained'] = True
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_name'] = model_choice
        st.session_state['metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Lấy thông tin mức độ quan trọng của đặc trưng đối với các mô hình cây
        if hasattr(model, 'feature_importances_'):
            st.session_state['feature_importances'] = model.feature_importances_
        else:
            st.session_state['feature_importances'] = None
            
    st.success(f"🎉 Huấn luyện thành công mô hình **{model_choice}**! Chuyển qua các Tab bên dưới để xem kết quả chi tiết.")

# ==========================================
# KHỞI TẠO CÁC CỬA SỔ TAB CHỨC NĂNG
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Đánh giá mô hình", 
    "🔮 Hệ thống Dự báo rủi ro"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab1:
    st.subheader("📋 Phân tích thông tin tập dữ liệu thô")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Tổng số dòng dữ liệu (Rows)", value=f"{df_raw.shape[0]:,}")
    with col2:
        st.metric(label="Tổng số cột đặc trưng (Columns)", value=df_raw.shape[1])
    with col3:
        # Tính dung lượng file ước lượng dựa trên DataFrame Memory
        file_size_mb = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric(label="Dung lượng bộ nhớ ước tính", value=f"{file_size_mb:.2f} MB")
        
    st.write("📂 **Xem trước 5 hàng dữ liệu đầu tiên (Head):**")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    st.write("📊 **Thống kê mô tả các biến đặc trưng đưa vào mô hình:**")
    # Chỉ thống kê mô tả các trường được dùng trực tiếp cho mô hình AI
    st.dataframe(df_raw[features + [target]].describe().T, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab2:
    st.subheader("🎯 Biểu đồ phân tích và phân phối phân loại")
    
    # 1. Trực quan hóa biến mục tiêu chính trước tiên
    target_counts = df_raw[target].value_counts().reset_index()
    target_counts.columns = ['Trạng thái', 'Số lượng']
    target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: 'Bình thường (0)', 1: 'Rủi ro/Gian lận (1)'})
    
    fig_target = px.bar(
        target_counts, x='Trạng thái', y='Số lượng',
        color='Trạng thái', text_auto=True,
        title="Biểu đồ phân phối Biến Mục Tiêu (default)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.divider()
    st.write("🔍 **Khảo sát phân phối của các biến đặc trưng đầu vào (X):**")
    
    # Cho phép người dùng chọn tối đa các biến muốn vẽ để giao diện không bị quá tải
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng để hiển thị biểu đồ phân phối:",
        options=features,
        default=features[:4],
        max_selections=8
    )
    
    if selected_features:
        # Bố trí biểu đồ lưới dạng 2 cột song song cân bằng hình học
        cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            current_col = cols[idx % 2]
            with current_col:
                fig_hist = px.histogram(
                    df_raw, x=feat, color=target,
                    marginal="box", barmode="overlay",
                    title=f"Phân phối biến {feat} theo lớp mục tiêu",
                    color_discrete_sequence=['#2b5c8f', '#d9534f'],
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến để theo dõi biểu đồ.")

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN"
# ------------------------------------------
with tab3:
    st.subheader("🏆 Đo lường hiệu suất và kiểm định chất lượng mô hình")
    
    # Kiểm tra điều phối trạng thái rỗng nếu chưa bấm Train
    if 'trained' not in st.session_state:
        st.info("ℹ️ Mô hình chưa được kích hoạt huấn luyện trên tập dữ liệu. Vui lòng chuyển qua thanh Sidebar bên trái và bấm nút '**Huấn luyện Mô hình**'.")
    else:
        metrics = st.session_state['metrics']
        
        # Hiển thị các chỉ tiêu định lượng vô hướng dạng thẻ Metric lớn
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="Độ chính xác (Accuracy)", value=f"{metrics['accuracy']:.4f}")
        with m_col2:
            st.metric(label="Độ chuẩn xác (Precision)", value=f"{metrics['precision']:.4f}", help="Tỷ lệ dự báo đúng gian lận trên tổng số ca hệ thống cảnh báo.")
        with m_col3:
            st.metric(label="Độ bao phủ (Recall)", value=f"{metrics['recall']:.4f}", help="Tỷ lệ phát hiện được hành vi gian lận trên tổng số ca gian lận thực tế xảy ra.")
        with m_col4:
            st.metric(label="Điểm số F1-Score", value=f"{metrics['f1']:.4f}")
            
        st.divider()
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.write("📊 **Ma trận nhầm lẫn (Confusion Matrix):**")
            cm = metrics['confusion_matrix']
            # Chuyển đổi dữ liệu ma trận nhầm lẫn thành dạng bảng nhiệt đồ dễ nhìn qua Plotly
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Nhãn Dự Báo", y="Nhãn Thực Tế", color="Số lượng"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                text_auto=True,
                color_continuous_scale='Blues',
                width=450, height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_right:
            st.write("📋 **Báo cáo chi tiết phân loại (Classification Report):**")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Purples', subset=['precision', 'recall', 'f1-score']), use_container_width=True)

        # Trực quan hóa tầm quan trọng của các biến đầu vào (Feature Importances) nếu dùng thuật toán Cây quyết định / Rừng cây
        if st.session_state['feature_importances'] is not None:
            st.divider()
            st.write("🎯 **Mức độ ảnh hưởng/quan trọng của các biến đối với quyết định phân loại:**")
            imp_df = pd.DataFrame({
                'Đặc trưng': features,
                'Mức độ quan trọng': st.session_state['feature_importances']
            }).sort_values(by='Mức độ quan trọng', ascending=True)
            
            fig_imp = px.bar(
                imp_df, x='Mức độ quan trọng', y='Đặc trưng', orientation='h',
                title=f"Độ quan trọng đặc trưng trong mô hình {st.session_state['model_name']}",
                color='Mức độ quan trọng', color_continuous_scale='Viridis',
                height=500
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab4:
    st.subheader("🔮 Ứng dụng dự báo nhanh rủi ro giao dịch tài chính")
    
    if 'trained' not in st.session_state:
        st.info("ℹ️ Vui lòng huấn luyện mô hình thành công tại thanh bên trước khi thực hiện chức năng dự báo rủi ro.")
    else:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        
        mode = st.radio(
            "Phương thức nhập dữ liệu đầu vào cần dự báo:",
            options=["Nhập chỉ số trực tiếp (Single)", "Tải tệp danh sách hàng loạt (Batch Processing)"],
            horizontal=True
        )
        
        # -------------------------------------------
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP
        # -------------------------------------------
        if mode == "Nhập chỉ số trực tiếp (Single)":
            st.write("📝 *Điền thông tin các chỉ số đặc trưng cần đánh giá rủi ro:*")
            
            # Sử dụng st.form để bao gói toàn bộ form nhập liệu, tránh việc reload trang liên tục khi gõ số
            with st.form("single_prediction_form"):
                # Tạo lưới các ô nhập liệu số tự động dựa trên biên min/max thực tế dữ liệu huấn luyện đầu vào
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(features):
                    col_idx = idx % 3
                    # Lấy giá trị mặc định là trung vị (median) của trường đó tránh lỗi rỗng
                    default_val = float(df_raw[feat].median())
                    min_val = float(df_raw[feat].min())
                    max_val = float(df_raw[feat].max())
                    
                    with form_cols[col_idx]:
                        input_data[feat] = st.number_input(
                            f"Thông số {feat}",
                            min_value=min_val * 3.0, # Mở rộng khoảng nhập biên phòng vệ ngoại lai
                            max_value=max_val * 3.0,
                            value=default_val,
                            format="%.4f"
                        )
                
                submit_predict = st.form_submit_button("🔍 Tiến hành phân tích & Dự báo", type="primary", use_container_width=True)
                
            if submit_predict:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame chuẩn hóa cấu trúc mẫu
                input_df = pd.DataFrame([input_data])
                
                # Áp dụng ĐÚNG bộ tiền xử lý StandardScaler đã fit ở lúc train
                input_scaled = scaler.transform(input_df[features])
                
                # Tiến hành dự đoán nhãn và xác suất rủi ro
                pred_class = model.predict(input_scaled)[0]
                
                st.divider()
                st.subheader("📊 Kết quả thẩm định rủi ro:")
                
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(input_scaled)[0]
                    prob_risk = pred_proba[1] * 100
                    
                    # Trực quan hóa thanh điểm phần trăm rủi ro nguy hiểm
                    st.write(f"**Tỷ lệ xác suất giao dịch gian lận:** `{prob_risk:.2f}%`")
                    st.progress(prob_risk / 100.0)
                
                if pred_class == 1:
                    st.error("🚨 **CẢNH BÁO NGUY HIỂM: Hệ thống phát hiện dấu hiệu GIAO DỊCH GIAN LẬN / RỦI RO CAO!**")
                else:
                    st.success("✅ **AN TOÀN: Giao dịch được thẩm định ở mức Bình thường, không phát hiện bất thường đáng ngại.**")

        # -------------------------------------------
        # CHẾ ĐỘ 2 — TẢI FILE THEO CẤU TRÚC DANH SÁCH HÀNG LOẠT
        # -------------------------------------------
        elif mode == "Tải tệp danh sách hàng loạt (Batch Processing)":
            st.write("📂 *Tải lên file định dạng Excel hoặc CSV có chứa đủ cấu trúc 14 cột đặc trưng đặc tả (`X_1` đến `X_14`):*")
            
            batch_file = st.file_uploader("Tải lên file danh sách khách hàng cần chấm điểm rủi ro", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra xem file test đầu vào có đủ cấu trúc cột hay không
                    missing_cols = [col for col in features if col not in df_batch.columns]
                    
                    if missing_cols:
                        st.error(f"Tệp tin tải lên bị thiếu các cột đặc trưng bắt buộc sau: {missing_cols}")
                    else:
                        st.info(f"Đọc thành công danh sách gồm {df_batch.shape[0]} bản ghi giao dịch cần chấm điểm.")
                        
                        # Thực hiện chuẩn hóa và dự đoán hàng loạt bằng mô hình trong session_state
                        batch_x_scaled = scaler.transform(df_batch[features])
                        batch_preds = model.predict(batch_x_scaled)
                        
                        # Gán kết quả đầu ra trực tiếp vào bảng hiển thị cho người dùng
                        df_result = df_batch.copy()
                        df_result['Dự báo (default)'] = batch_preds
                        df_result['Trạng thái rủi ro'] = df_result['Dự báo (default)'].map({0: 'Bình thường', 1: 'CẢNH BÁO GIAN LẬN'})
                        
                        if hasattr(model, "predict_proba"):
                            batch_probas = model.predict_proba(batch_x_scaled)[:, 1]
                            df_result['Xác suất rủi ro (%)'] = np.round(batch_probas * 100, 2)
                            
                        st.write("📊 **Bảng kết quả chấm điểm rủi ro tự động:**")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Tạo nút download kết quả đầu ra xuất thành file CSV dạng nén UTF-8-sig chống lỗi font tiếng Việt
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_output = csv_buffer.getvalue().encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả dự báo (.CSV)",
                            data=csv_output,
                            file_name="Ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
