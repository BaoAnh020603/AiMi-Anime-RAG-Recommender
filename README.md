# 🚀 AiMi Anime Recommender

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A semantic search and recommendation engine built on Streamlit, designed to find anime based on natural language descriptions, leveraging the power of Vector Embeddings and FAISS indexing.

## ✨ Features

- **🔍 Semantic Search**: Find anime by describing plots, themes, or moods (e.g., "dark fantasy with tragic character arcs")
- **🤖 RAG Architecture**: Utilizes a Retrieval-Augmented Generation (RAG) approach for highly accurate context matching
- **📊 Comprehensive Dataset**: Based on a curated sample of 500 anime metadata entries (1917–2025)
- **☁️ Public Deployment Ready**: Configured to automatically download data from Google Drive for easy public deployment

## 🔗 Live Demo

> **Note**: Replace this placeholder with your public URL after successful deployment.

[🌐 Try the Live Application](#)

## 🧠 Technical Overview

### How It Works

1. **Data Preparation**: Core metadata (Title, Studio, Tags, Synopsis) is combined into a comprehensive `rag_context` string for each anime entry
2. **Embedding**: The `rag_context` strings are encoded into 768-dimensional vectors using the Nomic v1.5 model
3. **Indexing**: Vectors are stored in a high-speed FAISS Index for fast nearest-neighbor search
4. **Query**: User queries are vectorized and matched against the FAISS Index to retrieve semantically similar anime

### Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Frontend/App Framework | `streamlit` | User Interface & Application Logic |
| Embedding Model | `nomic-ai/nomic-embed-text-v1.5` | Converts text to high-quality vectors |
| Vector Indexing | `faiss-cpu` | Optimized vector search and retrieval |
| Data & I/O | `pandas`, `gdown` | Handles Parquet data and downloads files from Google Drive |

## 🛠️ Local Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Internet connection (for downloading models and data)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/BaoAnh020603/AiMi-Anime-RAG-Recommender.git
cd AiMi-Anime-RAG-Recommender
```

2. **Create and activate virtual environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

> **⏳ First Run Note**: The initial launch will take several minutes to download the model and generate the FAISS index.

## 📁 Project Structure

```
AiMi-Anime-RAG-Recommender/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## 🚀 Deployment

This project is configured for easy deployment on [Streamlit Cloud](https://streamlit.io/cloud):

1. Fork or push this repository to GitHub
2. Connect your GitHub account to Streamlit Cloud
3. Deploy directly from the repository
4. The app automatically downloads required data files using `gdown`

The `.gitignore` prevents large files from being committed, while `app.py` uses a hardcoded Google Drive ID (`16bdNhA2DCgRevE3ZtaQIIym_lSRYqQO`) to dynamically retrieve data.

## 📊 Dataset

The application uses a curated dataset of 500 anime entries spanning from 1917 to 2025, stored in Parquet format with pre-computed embeddings for efficient retrieval.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Nomic AI](https://www.nomic.ai/) for the embedding model
- [FAISS](https://github.com/facebookresearch/faiss) by Meta Research
- [Streamlit](https://streamlit.io/) for the amazing framework

---

<div align="center">
Made with ❤️ by BaoAnh020603
</div>
