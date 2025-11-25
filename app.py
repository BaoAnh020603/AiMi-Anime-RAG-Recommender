import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import gdown # Cần thư viện này để tải file từ Google Drive

# TÊN FILE VÀ FILE ID
DATA_FILE = 'anime_dataset_small_nomic.parquet'
# !! QUAN TRỌNG: BẠN PHẢI THAY THẾ ID NÀY BẰNG ID FILE CỦA BẠN TỪ GOOGLE DRIVE !!
DATA_FILE_ID = 'https://drive.google.com/file/d/16bdNhA2DCgRevE3ZtaQIIym_lSRYVqQO/view?usp=sharing' 
MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'

# --- HÀM TẢI FILE NẶNG (SỬ DỤNG CACHE) ---
@st.cache_resource
def load_data_and_initialize_rag():
    st.info(f"Bắt đầu: Tải và Khởi tạo Hệ thống RAG...")
    
    # 1. TẢI FILE DỮ LIỆU TỪ GOOGLE DRIVE NẾU CHƯA TỒN TẠI
    if not os.path.exists(DATA_FILE):
        if DATA_FILE_ID == 'https://drive.google.com/file/d/16bdNhA2DCgRevE3ZtaQIIym_lSRYVqQO/view?usp=sharing':
            st.error("LỖI TRIỂN KHAI: Bạn chưa thay thế DATA_FILE_ID bằng ID file Google Drive của mình.")
            return None, None, None
            
        st.info(f"Đang tải file data lớn từ Google Drive (ID: {DATA_FILE_ID})...")
        try:
            # Gdown sẽ tải file và lưu với tên DATA_FILE
            gdown.download(id=DATA_FILE_ID, output=DATA_FILE, quiet=False, fuzzy=True)
            st.success("Tải file data thành công!")
        except Exception as e:
            st.error(f"LỖI TẢI FILE: Không thể tải file từ Google Drive. Đảm bảo ID và quyền chia sẻ công khai là đúng. Lỗi: {e}")
            return None, None, None
    else:
        st.info("File data đã tồn tại, tiến hành đọc file.")
    
    # 2. ĐỌC DỮ LIỆU
    try:
        df = pd.read_parquet(DATA_FILE)
    except Exception as e:
        st.error(f"LỖI ĐỌC FILE: Không thể đọc file Parquet. Lỗi: {e}")
        return None, None, None

    # 3. TẠO TRƯỜNG CONTEXT RAG
    st.info("Bước 1: Tạo trường 'rag_context'...")
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
    st.info(f"Bước 2: Tải mô hình Embedding: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        st.error(f"LỖI: Không thể tải mô hình embedding. Vui lòng kiểm tra kết nối internet. Lỗi chi tiết: {e}")
        return None, None, None
    
    # 5. Tạo Embeddings và Index FAISS
    st.info("Bước 3: Tạo Embeddings và Index FAISS...")
    embedding_texts = df['rag_context'].tolist()
    
    embeddings = model.encode(embedding_texts, show_progress_bar=False)
    
    # Tạo Index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    st.success(f"Khởi tạo RAG thành công! Tổng số entries: {len(df)}")
    return df, model, index

# --- 2. Hàm Tìm kiếm Ngữ nghĩa ---
def semantic_search(query: str, df: pd.DataFrame, model: SentenceTransformer, index: faiss.Index, k: int = 5):
    """Thực hiện tìm kiếm vector và trả về các anime phù hợp nhất."""
    
    # 2.1. Embed Query
    query_embedding = model.encode([query]) 
    
    # 2.2. Tìm kiếm trong Index FAISS
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # 2.3. Lấy kết quả từ DataFrame gốc
    results = df.iloc[indices[0]].copy()
    
    # Thêm khoảng cách L2 vào kết quả
    results['Distance'] = distances[0]
    
    return results.sort_values(by='Distance', ascending=True)

# --- 3. Giao diện Streamlit ---

st.title("🤖 Anime Recommender RAG (Public)") # Đổi tên cho bản Public

# Khởi tạo hệ thống
df, model, index = load_data_and_initialize_rag()

if df is not None:
    st.subheader("Hoàn tất Khởi tạo. Bây giờ bạn có thể tìm kiếm.")
    
    # Thanh tìm kiếm
    user_query = st.text_input(
        "Nhập truy vấn bằng ngôn ngữ tự nhiên:",
        "Dark fantasy anime with tragic character arcs and moral ambiguity"
    )
    
    k_recommendations = st.slider("Số lượng đề xuất:", 1, 10, 5)

    if user_query:
        start_time = time.time()
        
        # Thực hiện tìm kiếm
        with st.spinner("Đang tìm kiếm ngữ nghĩa..."):
            recommendations = semantic_search(user_query, df, model, index, k_recommendations)
        
        end_time = time.time()
        
        st.subheader(f"Top {k_recommendations} Đề xuất Anime:")
        st.write(f"*Tìm kiếm hoàn tất trong {end_time - start_time:.4f} giây.*")
        
        # Hiển thị kết quả
        for i, row in recommendations.iterrows():
            st.markdown("---")
            main_title = row.get('Main Title', 'N/A')
            official_en = row.get('Official Title (en)', 'N/A')
            max_rating = row.get('Max Rating', 0.0)
            filter_year = int(row.get('filter_year', 0))
            animation_work = row.get('Animation Work', 'N/A')
            synopsis = row.get('Synopsis', 'Không có tóm tắt')
            tags_content = row.get('Tags', 'Không có thẻ')
            
            st.markdown(f"**{main_title}** (Official EN: {official_en})")
            st.markdown(f"**Rating:** {max_rating:.2f} | **Năm:** {filter_year} | **Studio:** {animation_work}")
            st.markdown(f"**Tags:** *{tags_content}*")
            st.markdown(f"**Synopsis:** {synopsis}")
            st.caption(f"Độ gần (L2 Distance): {row['Distance']:.4f}")