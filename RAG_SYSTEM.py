import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import uuid

load_dotenv()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class SimpleModelSelector:
    """Simple class to handle model selection"""
    def __init__(self):
        # available llm models
        self.llm_models = {"openai" : "gpt-4.1-nano", "ollama" : "gemma3"}

        # Available embedding models with there dimensions
        self.embedding_models = {
            "openai":{
                "name" :"openAI Embeddings",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small"
            },
            "chroma":{
                "name" :"Chroma Embeddings",
                "dimensions": 384,
                "model_name": None
            },
            "nomic":{
                "name" :"Nomic Embeddings",
                "dimensions": 768,
                "model_name": "nomic-embed-text"
            }
        }

    def select_model(self):
        """Select a model from the available ones"""
        st.sidebar.title(" Model Selection")

        # Select LLM
        llm = st.sidebar.radio(
            "Select LLM Model",
            options=list(self.llm_models.keys()),
            format_func= lambda x: self.llm_models[x]
        )

        # Select Embeddings
        embeddings = st.sidebar.radio(
            "Select Embeddings Model",
            options=list(self.embedding_models.keys()),
            format_func= lambda x: self.embedding_models[x]['name']
        )
        return llm, embeddings
    

class SimplePDFProcessor:
    """Simple class to handle pdf processing"""
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    
    def read_pdf(self, pdf_file):
        """Read a pdf file """
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def create_chunks(self, text, pdf_file):
        """Create chunks of text from a pdf file"""
        chunks =[]
        start = 0

        while start < len(text):
            end = start +self.chunk_size

            if start > 0:
                start = start - self.chunk_overlap

            chunk = text[start:end]
            chunks.append({
                'id':str(uuid.uuid4()),
                'text':chunk,
                'metadata': {"source":pdf_file.name}
            })
            start = end
        return chunks


class SimpleRAGSystem:
    """Simple class to handle RAG system"""
    def __init__(self, embedding_model="openai", llm_model='openai'):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        # initialize chromadb
        self.db = chromadb.PersistentClient(path='./chroma_db')

        # setup embedding function based on model
        self.setup_embedding_function()
        # setup llm function based on model
        if llm_model == "openai":
            self.llm = OpenAI()
        else:
            self.llm = OpenAI(base_url="http://localhost:11433/v1", api_key="ollama")

        # get or create collections with proper handling
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        """Setup embedding function based on model"""
        try:
            if self.embedding_model == "openai":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
            elif self.embedding_model == "nomic":
                    # For Nomic embeddings via Ollama
                    self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                        api_key="ollama",
                        api_base="http://localhost:11434/v1",
                        model_name="nomic-embed-text",
                    )
            else:
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e
        
    def setup_collection(self):
        collection_name = f"document_{self.embedding_model}"
        try:
            collection = self.db.get_or_create_collection(
                name = collection_name,
                embedding_function=self.embedding_fn
            )
            st.info(
                f"Using collection for {self.embedding_model} embeddings"
            )
            return collection
        except Exception as e:
            st.error(f"Error Setting up collection: {str(e)}")
            raise e
        
    def add_documents(self, chunks):
        """Add documents to ChromaDB"""
        try:
            # Ensure collection exists
            if not self.collection:
                self.collection = self.setup_collection()

            # Add documents
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
        
    def query_documents(self, query, n_result=3):
        """query documents in chroma db"""
        try:
            #ensure collection exists
            if not self.collection:
                raise ValueError("No collection available")
            result = self.collection.query(query_texts=[query], n_results=n_result)
            return result
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None
        
    def generate_response(self, query, context):
        """generate response using llm"""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini" if self.llm_model == "openai" else "gemma3",
                messages=[
                    {"role":"system", "content":"you are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None
        
    def get_embedding_info(self):
        """get embedding info"""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info['name'],
            "dimensions":model_info['dimensions'],
            "model": self.embedding_model,
        }


def main(): 
    st.title("🤖 Query Documents using RAG ")

    # initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    #initialize model selector
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_model()

    # check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear() # clear processed files
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None # reset RAG system
        st.write("Embedding model changed. please re-upload you pdf file.")

    # initialize RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

            # Display current embedding model info
            embedding_info = st.session_state.rag_system.get_embedding_info()
            st.sidebar.info(
                f"Current embedding model: \n {embedding_info['name']}\n ({embedding_info['dimensions']})"
            )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return
    
    #file upload
    pdf_file = st.file_uploader("Upload PDF", type='pdf')

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        # process pdf file
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF...."):
            try:
                #extract text
                text = processor.read_pdf(pdf_file)
                #create chunks
                chunks = processor.create_chunks(text, pdf_file)
                #add to database
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # query interface
    if st.session_state.processed_files:
        st.markdown("-----")
        st.subheader("🔍 query ")
        query = st.text_input("Enter your query")

        if query:
            with st.spinner("Processing query...."):
                # get relevant chunks
                result = st.session_state.rag_system.query_documents(query)
                if result and result['documents']:
                    # generate response
                    response = st.session_state.rag_system.generate_response(
                        query,result['documents'][0]
                    )
                    if response:
                        #display results
                        st.markdown("### 📝 answer:")
                        st.write(response)

                        with st.expander("view Source Passages"):
                            for idx, doc in enumerate(result['documents'][0], 1):
                                st.markdown(f"**passage {idx}: **")
                                st.info(doc)
    else:
        st.info("👆 plz upload a pdf document to get started!")


if __name__ == "__main__":
    main()