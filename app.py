import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import gdown 

# CẤU HÌNH DATA VÀ MODEL (ID GOOGLE DRIVE CỦA BẠN ĐÃ ĐƯỢC DÁN VÀO ĐÂY)
DATA_FILE = 'anime_dataset_small_nomic.parquet'
DATA_FILE_ID = '16bdNhA2DCgRevE3ZtaQIIym_lSRYVqQO' 
MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'

# --- 1. Tải Dữ liệu, Tạo Index và Model (Chỉ chạy 1 lần) ---
@st.cache_resource
def load_data_and_initialize_rag():
    # 1. TẢI FILE DỮ LIỆU TỪ GOOGLE DRIVE NẾU CHƯA TỒN TẠI
    if not os.path.exists(DATA_FILE):
        try:
            # Gdown sẽ tải file và lưu với tên DATA_FILE
            gdown.download(id=DATA_FILE_ID, output=DATA_FILE, quiet=True, fuzzy=True)
        except Exception as e:
            st.error(f"LỖI TẢI DATA: Không thể tải file từ Google Drive. Vui lòng kiểm tra ID và quyền chia sẻ. Lỗi: {e}")
            return None, None, None
    
    # 2. ĐỌC DỮ LIỆU
    try:
        df = pd.read_parquet(DATA_FILE)
    except Exception as e:
        st.error(f"LỖI ĐỌC FILE: Không thể đọc file Parquet. Lỗi: {e}")
        return None, None, None

    # 3. TẠO TRƯỜNG CONTEXT RAG
    try:
        df['rag_context'] = (
            "Title: " + df['Main Title'].fillna('Unknown Title') + " | " +
            "Studio: " + df['Animation Work'].fillna('Unknown Studio') + " | " +
            "Tags: " + df['Tags'].fillna('No tags') + " | " + 
            "Synopsis: " + df['Synopsis'].fillna('No synopsis provided')
        )
    except KeyError as e:
        st.error(f"LỖỖI KEY: Cột {e} không tồn tại. Vui lòng kiểm tra lại chính tả tên cột.")
        return None, None, None

    # 4. Tải Mô hình Embedding 
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        st.error(f"LỖI: Không thể tải mô hình embedding. Vui lòng kiểm tra kết nối internet. Lỗi chi tiết: {e}")
        return None, None, None
    
    # 5. Tạo Embeddings và Index FAISS
    embedding_texts = df['rag_context'].tolist()
    embeddings = model.encode(embedding_texts, show_progress_bar=False)
    
    # Tạo Index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return df, model, index

# --- 2. Hàm Tìm kiếm Ngữ nghĩa ---
def semantic_search(query: str, df: pd.DataFrame, model: SentenceTransformer, index: faiss.Index, k: int = 5):
    """Thực hiện tìm kiếm vector và trả về các anime phù hợp nhất."""
    
    query_embedding = model.encode([query]) 
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    results = df.iloc[indices[0]].copy()
    results['Distance'] = distances[0]
    
    return results.sort_values(by='Distance', ascending=True)

# --- 3. Giao diện Streamlit ---

# Cấu hình trang (chế độ Wide)
st.set_page_config(
    page_title="AiMi Anime Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tiêu đề chính
st.markdown("<h1 style='text-align: center; color: #FF69B4;'>💖 AiMi Anime Recommender 🤖</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #808080;'>Tìm kiếm Anime bằng ngôn ngữ tự nhiên sử dụng Vector AI</h4>", unsafe_allow_html=True)


# Sử dụng st.spinner để ẩn các bước kỹ thuật
with st.spinner("🚀 Đang khởi động hệ thống Đề xuất AI... (Chỉ lần đầu tiên tải sẽ lâu)"):
    df, model, index = load_data_and_initialize_rag()

if df is not None:
    # Thay st.success bằng st.toast để thông báo gọn gàng hơn
    st.toast("Hệ thống AI đã sẵn sàng!")
    st.markdown("---")
    
    # CONTAINER CHO THANH TÌM KIẾM VÀ SLIDER
    search_container = st.container()
    with search_container:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "💬 Nhập mô tả Anime bạn muốn tìm:",
                "Dark fantasy anime with tragic character arcs and moral ambiguity",
                placeholder="Ví dụ: Slice of life comedy set in high school with healing atmosphere"
            )
        
        with col2:
            # Thay đổi nhãn slider
            k_recommendations = st.slider("Số lượng đề xuất:", 1, 10, 5, help="Chọn số lượng anime bạn muốn được đề xuất.")

    # KHỞI CHẠY TÌM KIẾM
    if user_query:
        start_time = time.time()
        
        # Thực hiện tìm kiếm
        with st.spinner(f"🔍 Đang tìm kiếm ngữ nghĩa cho '{user_query}'..."):
            recommendations = semantic_search(user_query, df, model, index, k_recommendations)
        
        end_time = time.time()
        
        # Tiêu đề kết quả
        st.markdown(f"## Top {k_recommendations} Đề xuất Phù hợp:")
        st.caption(f"🔎 Hoàn tất tìm kiếm trong {end_time - start_time:.4f} giây.")
        
        # HIỂN THỊ KẾT QUẢ DƯỚI DẠNG CARD
        for i, row in recommendations.iterrows():
            with st.container(border=True):
                main_title = row.get('Main Title', 'N/A')
                official_en = row.get('Official Title (en)', 'N/A')
                max_rating = row.get('Max Rating', 0.0)
                filter_year = int(row.get('filter_year', 0))
                animation_work = row.get('Animation Work', 'N/A')
                synopsis = row.get('Synopsis', 'Không có tóm tắt')
                tags_content = row.get('Tags', 'Không có thẻ')
                
                # Tính toán lại Similarity Score (Chuyển L2 Distance về điểm từ 0-1)
                # Max L2 Distance có thể khoảng 2.0. Chuẩn hóa về 0-100%
                # Dùng np.clip để tránh giá trị âm/lớn vô lý
                normalized_distance = np.clip(row['Distance'], 0, 1.5) 
                similarity_percentage = np.clip(100 - (normalized_distance * 100 / 1.5), 0, 100) # Chuẩn hóa dựa trên max distance 1.5
                
                col_info, col_rating = st.columns([3, 1])
                
                with col_info:
                    # Tiêu đề chính + Năm sản xuất
                    st.markdown(f"### 🏆 {main_title} *({official_en})*")
                    st.markdown(f"**🎬 Studio:** {animation_work} | **📅 Năm:** {filter_year}")
                    
                    # Rút gọn Tags nếu quá dài
                    if len(tags_content) > 150:
                        display_tags = tags_content[:150] + "..."
                    else:
                        display_tags = tags_content
                    
                    st.markdown(f"**🏷️ Thể loại chính:** *{display_tags}*")
                    st.markdown(f"**📖 Tóm tắt:** {synopsis}")
                
                with col_rating:
                    # Hiển thị Rating và Độ Tương đồng bằng st.metric
                    st.metric(label="⭐ Đánh giá (10)", value=f"{max_rating:.2f}")
                    # Hiển thị độ tương đồng dưới dạng %
                    st.metric(label="🎯 Độ tương đồng", value=f"{similarity_percentage:.1f}%")
