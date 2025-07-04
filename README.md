# CLIP-Based Image Search Engine

A powerful semantic image search engine that uses OpenAI's CLIP model and FAISS (Facebook AI Similarity Search) to find visually and semantically similar images from large datasets.

## 🚀 Features

- **Semantic Search**: Find images based on visual content and semantic understanding
- **CLIP Integration**: Utilizes OpenAI's CLIP model for robust image feature extraction
- **Efficient Indexing**: Uses FAISS for fast similarity search across large image collections
- **User-Friendly Interface**: Interactive Gradio web interface for easy image uploads and results viewing
- **Persistent Storage**: Saves and loads pre-computed indices for faster subsequent searches
- **GPU Acceleration**: Supports CUDA for faster processing when available

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Computer Vision**: CLIP (Contrastive Language-Image Pre-training)
- **Search Engine**: FAISS (Facebook AI Similarity Search)
- **Web Interface**: Gradio
- **Image Processing**: PIL, torchvision
- **Data Handling**: NumPy, tqdm

## 📋 Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional but recommended)
- Sufficient disk space for your image dataset and index files

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amirreza-al/visual_search_engine.git
   cd visual_search_engine
   ```

2. **Install required packages**
   ```bash
   pip install faiss-cpu gradio transformers torchvision torch pillow numpy tqdm
   ```

   For GPU support (recommended):
   ```bash
   pip install faiss-gpu gradio transformers torchvision torch pillow numpy tqdm
   ```

## 📁 Project Structure

```
visual_search_engine/
├── visual_search_engine.ipynb
├── Dataset/                 # Your image dataset folder
├── output/                   # Generated index and feature files
│   ├── faiss_index.bin
│   ├── features.npy
│   └── filenames.npy
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

1. **Prepare your image dataset**
   - Create a `Dataset` folder in the project directory
   - Add your images (supports .jpg, .jpeg, .png, .bmp formats)

2. **Run the application**
   ```bash
   jupyter notebook visual_search_engine.ipynb
   ```

3. **Build the index**
   - The first run will automatically extract features and build the FAISS index
   - This process may take some time depending on your dataset size

4. **Start searching**
   - Upload a query image through the Gradio interface
   - View the top 5 most similar images from your dataset

## 📊 How It Works

### 1. Feature Extraction
The system uses OpenAI's CLIP model to extract high-dimensional feature vectors from images:
- Images are processed through CLIP's vision transformer
- Features are L2-normalized for consistent similarity measurements
- Each image is represented as a 512-dimensional vector

### 2. Index Building
FAISS creates an efficient search index:
- Features are stored in a flat L2 distance index
- Index is persisted to disk for faster subsequent loads
- Filenames are mapped to feature vectors for result retrieval

### 3. Similarity Search
When querying with a new image:
- Query image features are extracted using the same CLIP model
- FAISS performs efficient k-nearest neighbor search
- Returns the most similar images based on cosine similarity

## ⚙️ Configuration

Key parameters you can modify:

```python
# Global Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = "Dataset"  # Your image dataset folder
INDEX_FILE = "output/faiss_index.bin"
FEATURES_FILE = "output/features.npy"
FILENAMES_FILE = "output/filenames.npy"
TOP_K = 5  # Number of results to return
```

## 🎯 Use Cases

- **E-commerce**: Find similar products in online catalogs
- **Digital Asset Management**: Organize and search large image libraries
- **Content Discovery**: Recommend visually similar content
- **Research**: Analyze visual patterns in image datasets
- **Art and Design**: Find inspiration through visual similarity

## 🔄 Performance Optimization

- **GPU Usage**: The system automatically uses CUDA when available
- **Batch Processing**: Features are extracted efficiently using vectorized operations
- **Index Persistence**: Pre-computed indices are saved to avoid reprocessing
- **Memory Management**: Features are stored as float32 for optimal memory usage

## 📈 Scalability

The system can handle large image datasets efficiently:
- FAISS provides logarithmic search complexity
- Index building is parallelizable
- Memory usage scales linearly with dataset size
- GPU acceleration significantly improves processing speed

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:
- Report bugs and suggest features
- Improve documentation
- Add support for additional image formats
- Implement new search algorithms
- Optimize performance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Facebook AI Research for FAISS
- Hugging Face for the Transformers library
- The Gradio team for the web interface framework

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation for troubleshooting tips
- Review the code comments for implementation details

---

**Built with ❤️ by [Amirreza Alipour](https://github.com/Amirreza-al)**
