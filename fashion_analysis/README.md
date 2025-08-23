A complete multimodal Retrieval-Augmented Generation (RAG) pipeline for fashion analysis

- Implement a complete multimodal RAG pipeline integrating computer vision with structured fashion datasets

- Use image encoding techniques for vector-based similarity matching and retrieval

- Explain how to enhance LLM responses by augmenting them with retrieved contextual information

- Develop a user-friendly interface with Gradio

- Apply modular programming principles to create maintainable AI applications

- Gain practical experience with state-of-the-art multimodal AI techniques



This project is using Meta's Llama 3.2 90B Vision Instruct model, this system identifies clothing items, retrieves relevant metadata, and provides actionable insights based on visual inputs.

Upload any fashion photo, and Style Finder will identify garments, analyze their style elements, and provide detailed information about each item. It can also find similar items at different price points, making high-end fashion more accessible.


### Setting up your development environment
```
git clone --no-checkout https://github.com/HaileyTQuach/style-finder.git
cd style-finder
git checkout 1-start

python3.11 -m venv venv
source venv/bin/activate # activate venv
pip install -r requirements.txt
```

Download the dataset
```
wget -O swift-style-embeddings.pkl https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-with-embeddings.pkl
```