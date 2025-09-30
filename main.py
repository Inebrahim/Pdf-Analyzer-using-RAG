import os
import re
import gradio as gr
import pymupdf
import networkx as nx
import tempfile
import shutil
from PathRAG import PathRAG, QueryParam
from pyvis.network import Network

# --- Global State & Configuration ---
rag_instance = None
# Permanent directory for uploaded PDFs
DOCUMENTS_DIR = "documents"
# Permanent directory for the RAG data itself
PERMANENT_DATA_DIR = "autosar_rag_data"

def initialize_instance():
    """
    Initializes the RAG instance ONCE at startup, loading from the permanent directory.
    """
    global rag_instance
    if rag_instance is None:
        if not os.path.exists(PERMANENT_DATA_DIR):
            os.makedirs(PERMANENT_DATA_DIR)
        
        print(f"Initializing PathRAG, loading data from: '{PERMANENT_DATA_DIR}'")
        rag_instance = PathRAG(working_dir=PERMANENT_DATA_DIR)
        print("PathRAG instance is ready.")
    return rag_instance

def find_pdfs_in_repo():
    """Finds all PDF files in the permanent DOCUMENTS_DIR."""
    if not os.path.exists(DOCUMENTS_DIR):
        return []
    return sorted([f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")])

# --- Gradio Callback Functions ---

def save_uploaded_pdf(pdf_file_obj):
    """
    This function's ONLY job is to save the uploaded file to the permanent
    'documents' folder. It is simple and robust.
    """
    if not pdf_file_obj:
        return "No file uploaded.", gr.Dropdown(choices=find_pdfs_in_repo())
    
    temp_path = pdf_file_obj.name
    filename = os.path.basename(temp_path)
    permanent_path = os.path.join(DOCUMENTS_DIR, filename)
    
    print(f"Saving uploaded file to permanent storage: {permanent_path}")
    shutil.copyfile(temp_path, permanent_path)
    
    # After saving, refresh the list of available PDFs for the dropdown
    new_pdf_list = find_pdfs_in_repo()
    
    return f"'{filename}' uploaded successfully.", gr.Dropdown(choices=new_pdf_list, value=filename)

def wipe_and_index_pdf(pdf_filename: str, progress=gr.Progress(track_tqdm=True)):
    """
    WIPES all previous data and then indexes the selected PDF from the 'documents' directory.
    """
    global rag_instance
    if not pdf_filename:
        return "Error: No PDF selected from the dropdown."
        
    pdf_path = os.path.join(DOCUMENTS_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        return f"Error: File '{pdf_filename}' not found. It may have been deleted."

    try:
        # WIPE: Delete the permanent data directory and re-initialize the instance
        print(f"Wiping all data in '{PERMANENT_DATA_DIR}'...")
        if os.path.exists(PERMANENT_DATA_DIR):
            shutil.rmtree(PERMANENT_DATA_DIR)
        rag_instance = None # Force re-initialization
        initialize_instance()
        
        print(f"Opening PDF for indexing from permanent path: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        all_content = ["".join(page.get_text() for page in doc)]
        doc.close()
        
        if not all_content[0].strip():
            return "The selected PDF contains no extractable text."

        print("Inserting document into PathRAG...")
        rag_instance.insert(all_content)
        
        return f"Successfully wiped all data and indexed '{pdf_filename}'. The system is now ready."
    except Exception as e:
        print(f"Error during indexing: {e}"); import traceback; traceback.print_exc()
        return f"Failed to index PDF. Error: {e}"

def answer_question(question: str):
    """Answers a question based on the indexed knowledge."""
    global rag_instance
    if not rag_instance: return "Error: Please index a document first."
    if not question or not question.strip(): return "Please enter a question."
    try:
        print(f"Querying with: '{question}'")
        params = QueryParam()
        response = rag_instance.query(question, param=params)
        return response
    except Exception as e:
        print(f"Error during query: {e}"); import traceback; traceback.print_exc()
        return f"Failed to get answer. Error: {e}"

# --- Gradio UI with the new, robust workflow ---
def create_gradio_ui():
    pdf_choices = find_pdfs_in_repo()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Document RAG System") as demo:
        with gr.Tabs():
            with gr.TabItem("Q&A"):
                gr.Markdown("# Document Analysis with PathRAG")
                gr.Markdown("Ask questions about the currently indexed document(s).")
                question_box = gr.Textbox(label="Ask a Question", placeholder="e.g., What is the scope of this document?", lines=3)

                ask_btn = gr.Button("Ask", variant="primary")
                answer_box = gr.Textbox(label="Answer", interactive=False, lines=20)   
                ask_btn.click(fn=answer_question, inputs=question_box, outputs=answer_box)
                
            with gr.TabItem("Admin & Indexing"):
                gr.Markdown("# Indexing and Data Management")
                status_box = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("## Step 1: Upload a New Document")
                gr.Markdown("Upload a new PDF. This will save it to the server's permanent 'documents' folder.")
                pdf_upload_admin = gr.File(label="Upload PDF", file_types=[".pdf"])

                gr.Markdown("## Step 2: Select and Index Document")
                gr.Markdown("After uploading, select the PDF from the dropdown. **Indexing will wipe all previously indexed data.**")
                pdf_dropdown_admin = gr.Dropdown(label="Select PDF to Index", choices=pdf_choices, value=pdf_choices[0] if pdf_choices else None)
                index_btn = gr.Button("Wipe and Start Indexing", variant="primary")

                # Connect the upload component to the save function
                pdf_upload_admin.upload(fn=save_uploaded_pdf, inputs=pdf_upload_admin, outputs=[status_box, pdf_dropdown_admin])
                # Connect the index button to the indexing function
                index_btn.click(fn=wipe_and_index_pdf, inputs=pdf_dropdown_admin, outputs=status_box)

    return demo

if __name__ == "__main__":
    # Ensure the permanent documents directory exists on startup
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        
    initialize_instance()
    app = create_gradio_ui()
    app.launch(share=True)