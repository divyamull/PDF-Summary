"""
Document Summarization Streamlit App

This Streamlit application provides a web interface for summarizing PDF and text documents.
It uses the BART-large-CNN model for text summarization and supports both PDF and TXT files.

Key Features:
- Interactive file upload widget
- PDF and TXT file processing
- Text extraction and preview
- Automatic text summarization
- Progress indicators
- Error handling and validation
- Download summary functionality

Dependencies:
- streamlit: Web app framework
- transformers: Hugging Face transformers for summarization
- pdfplumber: PDF text extraction
- torch: PyTorch for model backend
"""

import streamlit as st
from transformers import pipeline
import pdfplumber
import os
import tempfile
from typing import Optional
import time

# Application Configuration
MAX_TEXT_LENGTH = 1024  # Maximum number of characters to process
MAX_SUMMARY_LENGTH = 130  # Maximum length of generated summary
MIN_SUMMARY_LENGTH = 30  # Minimum length of generated summary
ALLOWED_EXTENSIONS = {'pdf', 'txt'}  # Supported file types

# Configure Streamlit page
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_summarization_model():
    """
    Load and cache the summarization model to avoid reloading on every interaction.
    
    Returns:
        pipeline: Initialized summarization pipeline
    """
    with st.spinner("Loading summarization model... This may take a moment."):
        return pipeline("summarization", model="facebook/bart-large-cnn")

def is_valid_file_extension(filename: str) -> bool:
    """
    Validate if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path: str, file_extension: str) -> Optional[str]:
    """
    Extract text content from the uploaded file based on its type.
    
    Args:
        file_path (str): Path to the temporary stored file
        file_extension (str): Extension of the file ('pdf' or 'txt')
        
    Returns:
        Optional[str]: Extracted text content or None if extraction fails
    """
    try:
        if file_extension == 'pdf':
            # Handle PDF files using pdfplumber
            with pdfplumber.open(file_path) as pdf_document:
                # Extract text from each page and join with spaces
                return " ".join(page.extract_text() or "" for page in pdf_document.pages).strip()
        else:  # txt file
            # Handle plain text files
            with open(file_path, 'r', encoding='utf-8') as text_file:
                return text_file.read().strip()
    except Exception as error:
        st.error(f"Error extracting text: {str(error)}")
        return None

def main():
    """Main Streamlit application function"""
    
    # App header
    st.title("📄 Document Summarizer")
    st.markdown("Upload a PDF or text document to get an AI-generated summary using BART-large-CNN model.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model information
        st.info("""
        **Model**: facebook/bart-large-cnn
        
        **Supported formats**: PDF, TXT
        
        **Max text length**: 1,024 characters
        """)
        
        # Advanced settings
        st.subheader("Advanced Settings")
        custom_max_length = st.slider("Max Summary Length", 50, 200, MAX_SUMMARY_LENGTH)
        custom_min_length = st.slider("Min Summary Length", 10, 50, MIN_SUMMARY_LENGTH)
        
        # Statistics placeholder
        if 'stats' in st.session_state:
            st.subheader("📊 Statistics")
            stats = st.session_state.stats
            st.metric("Original Length", f"{stats['original_length']} chars")
            st.metric("Summary Length", f"{stats['summary_length']} chars")
            st.metric("Compression Ratio", f"{stats['compression_ratio']:.1%}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Document")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload a document to summarize. Maximum text length processed: 1,024 characters."
        )
        
        if uploaded_file is not None:
            # Display file information
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📁 File size: {uploaded_file.size:,} bytes")
            
            # Process button
            if st.button("🚀 Generate Summary", type="primary"):
                process_document(uploaded_file, custom_max_length, custom_min_length)
    
    with col2:
        st.subheader("📋 Results")
        
        # Display results if available
        if 'summary_result' in st.session_state:
            result = st.session_state.summary_result
            
            # Summary output
            st.markdown("### 🎯 Generated Summary")
            st.markdown(f"*{result['summary']}*")
            
            # Download button for summary
            st.download_button(
                label="📥 Download Summary",
                data=result['summary'],
                file_name=f"summary_{uploaded_file.name if uploaded_file else 'document'}.txt",
                mime="text/plain"
            )
            
            # Show original text preview if available
            if 'original_text' in result:
                with st.expander("📖 Original Text Preview"):
                    st.text_area(
                        "Extracted Text",
                        value=result['original_text'][:500] + "..." if len(result['original_text']) > 500 else result['original_text'],
                        height=200,
                        disabled=True
                    )
        else:
            st.info("👆 Upload a document and click 'Generate Summary' to see results here.")

def process_document(uploaded_file, max_length: int, min_length: int):
    """
    Process the uploaded document and generate summary.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        max_length (int): Maximum summary length
        min_length (int): Minimum summary length
    """
    try:
        # Validate file extension
        if not is_valid_file_extension(uploaded_file.name):
            st.error(f"❌ Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Save uploaded file temporarily
        status_text.text("📁 Saving file...")
        progress_bar.progress(20)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Step 2: Extract text
            status_text.text("🔍 Extracting text...")
            progress_bar.progress(40)
            
            file_extension = uploaded_file.name.rsplit('.', 1)[1].lower()
            extracted_text = extract_text_from_file(tmp_file_path, file_extension)
            
            if not extracted_text:
                st.error("❌ No text could be extracted from the file")
                return
            
            # Step 3: Truncate text if necessary
            original_length = len(extracted_text)
            if len(extracted_text) > MAX_TEXT_LENGTH:
                extracted_text = extracted_text[:MAX_TEXT_LENGTH]
                st.warning(f"⚠️ Text truncated to {MAX_TEXT_LENGTH} characters for processing")
            
            # Step 4: Load model and generate summary
            status_text.text("🤖 Loading AI model...")
            progress_bar.progress(60)
            
            summarizer = load_summarization_model()
            
            status_text.text("✨ Generating summary...")
            progress_bar.progress(80)
            
            # Generate summary
            summary_result = summarizer(
                extracted_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            # Step 5: Store results
            progress_bar.progress(100)
            status_text.text("✅ Summary generated successfully!")
            
            # Calculate statistics
            summary_length = len(summary_result)
            compression_ratio = 1 - (summary_length / original_length)
            
            # Store in session state
            st.session_state.summary_result = {
                'summary': summary_result,
                'original_text': extracted_text,
                'original_length': original_length,
                'summary_length': summary_length
            }
            
            st.session_state.stats = {
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio
            }
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success(f"🎉 Summary generated successfully! Compressed from {original_length} to {summary_length} characters.")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    
    except Exception as error:
        st.error(f"❌ An error occurred while processing the file: {str(error)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

if __name__ == "__main__":
    main()
