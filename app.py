import streamlit as st
from transformers import pipeline
import pdfplumber
import os
import tempfile
from typing import Optional, List
import time

# Set page configuration
st.set_page_config(
    page_title="Fast Document Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

@st.cache_resource
def load_model(model_name: str):
    with st.spinner(f"Loading model: {model_name}"):
        return pipeline("summarization", model=model_name)

def is_valid_file_extension(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path: str, file_extension: str) -> Optional[str]:
    try:
        if file_extension == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                return " ".join(page.extract_text() or "" for page in pdf.pages).strip()
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def split_text_into_chunks(text: str, max_chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def process_document(uploaded_file, model_name: str, max_length: int, min_length: int, max_chunks: int, use_refined_summary: bool):
    try:
        if not is_valid_file_extension(uploaded_file.name):
            st.error(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("📁 Saving file...")
        progress_bar.progress(10)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # Extract text
            status_text.text("🔍 Extracting text...")
            progress_bar.progress(20)
            extension = uploaded_file.name.rsplit('.', 1)[1].lower()
            extracted_text = extract_text_from_file(tmp_path, extension)

            if not extracted_text:
                st.error("No text extracted.")
                return

            original_length = len(extracted_text)

            # Split into chunks
            status_text.text("✂️ Splitting text into chunks...")
            text_chunks = split_text_into_chunks(extracted_text)
            text_chunks = text_chunks[:max_chunks]  # Limit processing
            progress_bar.progress(30)

            summarizer = load_model(model_name)

            summaries = []
            for i, chunk in enumerate(text_chunks):
                status_text.text(f"🧠 Summarizing chunk {i+1}/{len(text_chunks)}...")
                summary = summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
                progress_bar.progress(30 + int(50 * (i + 1) / len(text_chunks)))

            full_chunk_summary = "\n\n".join(summaries)

            # Optional: Refined final summary
            if use_refined_summary and len(summaries) > 1:
                status_text.text("🧠 Generating high-level refined summary...")
                refined_summary = summarizer(
                    full_chunk_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                final_summary = refined_summary
            else:
                final_summary = full_chunk_summary

            summary_length = len(final_summary)
            compression_ratio = 1 - (summary_length / original_length)

            st.session_state.summary_result = {
                'summary': final_summary,
                'chunk_summaries': full_chunk_summary,
                'original_text': extracted_text,
                'original_length': original_length,
                'summary_length': summary_length
            }

            st.session_state.stats = {
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio
            }

            progress_bar.progress(100)
            status_text.text("✅ Summary generated!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Done! From {original_length} to {summary_length} characters.")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def main():
    st.title("⚡ Fast & Smart Document Summarizer")
    st.markdown("Upload a **PDF or TXT file** to generate a concise AI-powered summary using optimized settings.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        model_choice = st.selectbox("Choose model", ["t5-small", "facebook/bart-base", "facebook/bart-large-cnn"])
        max_length = st.slider("Max Summary Length", 50, 500, 250)
        min_length = st.slider("Min Summary Length", 20, 200, 100)
        max_chunks = st.slider("Max Chunks to Process", 1, 100, 20)
        use_refined_summary = st.checkbox("🧠 Generate final summary of summaries", value=True)

        if 'stats' in st.session_state:
            st.subheader("📊 Summary Stats")
            stats = st.session_state.stats
            st.metric("Original Length", f"{stats['original_length']} chars")
            st.metric("Summary Length", f"{stats['summary_length']} chars")
            st.metric("Compression Ratio", f"{stats['compression_ratio']:.1%}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload a document to summarize. Full document is processed in chunks."
        )

        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📁 Size: {uploaded_file.size:,} bytes")

            if st.button("🚀 Generate Summary"):
                process_document(uploaded_file, model_choice, max_length, min_length, max_chunks, use_refined_summary)

    with col2:
        st.subheader("📋 Summary Output")
        if 'summary_result' in st.session_state:
            result = st.session_state.summary_result

            st.markdown("### 🧠 Final Summary")
            st.markdown(f"*{result['summary']}*")

            st.download_button(
                label="📥 Download Summary",
                data=result['summary'],
                file_name=f"summary_{uploaded_file.name if uploaded_file else 'document'}.txt",
                mime="text/plain"
            )

            with st.expander("🧩 Chunk-wise Summaries"):
                st.text_area("All Chunks Summary", result['chunk_summaries'], height=300)

            with st.expander("📖 Original Text Preview"):
                st.text_area(
                    "Extracted Text",
                    value=result['original_text'][:500] + "..." if len(result['original_text']) > 500 else result['original_text'],
                    height=200,
                    disabled=True
                )
        else:
            st.info("👆 Upload a document and click 'Generate Summary' to see results here.")

if __name__ == "__main__":
    main()
