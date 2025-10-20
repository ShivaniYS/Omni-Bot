import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

st.set_page_config(
    page_title="OmniBot - Your AI Companion",
    page_icon="🤖",
    layout="wide"
)

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "OmniBot-Streamlit"))

if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'document_store' not in st.session_state:
    st.session_state.document_store = {}
if 'code_history' not in st.session_state:
    st.session_state.code_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Brainy Buddy"
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = not bool(groq_api_key)
if 'document_chat_history' not in st.session_state:
    st.session_state.document_chat_history = []

with st.sidebar:
    st.title("🤖 OmniBot Settings")
    
    st.subheader("🔑 API Status")
    if groq_api_key:
        st.success("✅ Groq API Key: Configured")
        st.session_state.demo_mode = False
    else:
        st.warning("🔶 Demo Mode - Using sample responses")
        st.session_state.demo_mode = True
    
    langsmith_status = "✅ Configured" if langchain_api_key else "❌ Not Configured"
    st.write(f"LangSmith Tracing: {langsmith_status}")
    
    st.subheader("🧠 Model Settings")
    mode = st.selectbox("Choose Your Assistant Mode", ["Brainy Buddy", "DocuMind", "CodeCraft"])
    st.session_state.current_mode = mode
    
    if groq_api_key:
        engine = st.selectbox("Select Groq Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 2000, 512, 50)
    else:
        engine = "llama-3.1-8b-instant"
        temperature = 0.7
        max_tokens = 512
        st.info("Add GROQ_API_KEY to unlock model settings")
    
    if mode == "DocuMind":
        session_id = st.text_input("Session ID for document chat", value="default_session")
    
    st.markdown("---")
    st.subheader("🔄 Clear History")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
        if st.button("Clear Code", use_container_width=True):
            st.session_state.code_history = []
            st.rerun()
    with col2:
        if st.button("Clear Docs", use_container_width=True):
            st.session_state.document_store = {}
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.document_chat_history = []
            st.rerun()

st.title("🤖 OmniBot - Your Intelligent AI Companion")

if st.session_state.demo_mode:
    st.warning("🔶 **Demo Mode** - Add GROQ_API_KEY to Streamlit secrets for full AI capabilities")
else:
    st.success("🚀 **Full AI Mode** - All features enabled!")

st.markdown("---")

def get_demo_response(mode, prompt, programming_language="Python", code_task="Write Code"):
    if mode == "Brainy Buddy":
        responses = [
            f"Hello! I'm Brainy Buddy in demo mode. You asked: '{prompt}'. Add GROQ_API_KEY for real AI responses.",
            f"Interesting question about '{prompt}'. Enable full features with Groq API key.",
            f"Demo response to: '{prompt}'. Add API key for intelligent answers."
        ]
        return responses[len(prompt) % 3]
    
    elif mode == "CodeCraft":
        code_responses = {
            "Write Code": f"""# Demo Response for {programming_language}
For: {prompt}

With Groq API, I would provide complete {programming_language} code with explanations.
Add GROQ_API_KEY to unlock real code generation.""",

            "Debug/Explain": f"""# Debugging Demo
For: {prompt}

Enable full mode for code analysis and debugging assistance.""",

            "Optimize": f"""# Optimization Demo
Task: {prompt}

Add API key for performance optimization and best practices."""
        }
        return code_responses.get(code_task, code_responses["Write Code"])
    
    elif mode == "DocuMind":
        return f"""📚 DocuMind Demo

Question: {prompt}

Add GROQ_API_KEY to process documents and get real answers from your files."""

def get_llm():
    if groq_api_key:
        try:
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=engine,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            st.error(f"Error initializing Groq LLM: {str(e)}")
            return None
    else:
        return None

def process_documents(uploaded_files):
    documents = []
    temp_files = []
    
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_files.append(temp_file.name)
            
            loader = PyPDFLoader(temp_file.name)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
            documents.extend(docs)
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            return vectorstore, splits
        return None, []
    
    except Exception as e:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise e

def get_document_answer(question, vectorstore, chat_history):
    llm = get_llm()
    if not llm:
        return "API key not configured"
    
    try:
        # Search for relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build conversation history
        history_text = ""
        for i, msg in enumerate(chat_history[-4:]):  # Last 4 messages for context
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        # Create prompt with context and history
        prompt_text = f"""You are DocuMind, an expert at analyzing documents. Use the provided context to answer the question.

Context from documents:
{context}

Conversation history:
{history_text}

Question: {question}

Provide a clear, accurate answer based only on the context. If the answer isn't in the context, say so."""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on provided documents."),
            ("human", "{question}"),
        ])
        
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"question": prompt_text})
        return response
        
    except Exception as e:
        return f"Error processing document question: {str(e)}"

if st.session_state.current_mode == "Brainy Buddy":
    st.header("💬 Brainy Buddy - Intelligent Conversations")
    
    if st.session_state.demo_mode:
        st.info("🔶 Demo Mode - Add GROQ_API_KEY for real AI responses")
    
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        llm = get_llm()
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not llm:
                with st.spinner("💭 Demo mode..."):
                    response = get_demo_response("Brainy Buddy", prompt)
                    st.markdown(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
            else:
                with st.spinner("🤔 Thinking..."):
                    try:
                        messages = [
                            ("system", "You are Brainy Buddy, a helpful AI assistant. Provide concise and informative responses."),
                            *[(msg["role"], msg["content"]) for msg in st.session_state.conversation_history[-6:]],
                        ]
                        prompt_template = ChatPromptTemplate.from_messages(messages)
                        chain = prompt_template | llm | StrOutputParser()
                        response = chain.invoke({"question": prompt})
                        st.markdown(response)
                        st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

elif st.session_state.current_mode == "DocuMind":
    st.header("📚 DocuMind - Document Intelligence")
    
    if st.session_state.demo_mode:
        st.info("🔶 Demo Mode - Add GROQ_API_KEY to process documents")
    else:
        st.markdown("Upload PDF documents and ask questions about their content")
    
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and not st.session_state.demo_mode:
        if st.session_state.processed_files:
            st.info(f"📁 Processed files: {', '.join(st.session_state.processed_files)}")
        
        if st.session_state.vectorstore is None or st.button("Reprocess Documents"):
            with st.spinner("📄 Processing documents... This may take a moment."):
                try:
                    vectorstore, splits = process_documents(uploaded_files)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success(f"✅ Processed {len(uploaded_files)} document(s) with {len(splits)} chunks! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Display document chat history
    if st.session_state.document_chat_history:
        st.subheader("Document Conversation")
        for message in st.session_state.document_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    user_question = st.text_input("Enter your question about the documents:", key="doc_question")
    
    if user_question:
        if st.session_state.demo_mode:
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                response = get_demo_response("DocuMind", user_question)
                st.markdown(response)
        else:
            if st.session_state.vectorstore is None:
                st.warning("Please upload and process documents first.")
            else:
                with st.spinner("🔍 Searching documents..."):
                    # Add user question to chat history
                    st.session_state.document_chat_history.append({"role": "user", "content": user_question})
                    
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    
                    with st.chat_message("assistant"):
                        try:
                            response = get_document_answer(user_question, st.session_state.vectorstore, st.session_state.document_chat_history)
                            st.markdown(response)
                            st.session_state.document_chat_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Error processing question: {str(e)}"
                            st.error(error_msg)
                            st.session_state.document_chat_history.append({"role": "assistant", "content": error_msg})

elif st.session_state.current_mode == "CodeCraft":
    st.header("💻 CodeCraft - Your Coding Assistant")
    
    if st.session_state.demo_mode:
        st.info("🔶 Demo Mode - Add GROQ_API_KEY for real code assistance")
    
    col1, col2 = st.columns(2)
    with col1:
        programming_language = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "SQL", "TypeScript", "HTML/CSS", "Other"])
    with col2:
        code_task = st.selectbox("Type of Assistance", ["Write Code", "Debug/Explain", "Optimize", "Learn Concepts", "Code Review"])
    
    if st.session_state.code_history:
        st.subheader("📚 Recent Code Interactions")
        for i, interaction in enumerate(reversed(st.session_state.code_history[-3:])):
            with st.expander(f"💻 {interaction['task']} - {interaction['language']}"):
                st.markdown("**Your Question/Code:**")
                st.code(interaction['question'], language=interaction.get('language', 'text').lower())
                st.markdown("**Response:**")
                st.markdown(interaction['response'])
    
    code_prompt = st.text_area("Describe your coding task or paste your code:", height=150, placeholder="e.g., Write a Python function to calculate factorial...")
    
    if st.button("Get Code Help 🚀", type="primary"):
        if code_prompt:
            with st.spinner("💻 Working on your code..."):
                if st.session_state.demo_mode:
                    response = get_demo_response("CodeCraft", code_prompt, programming_language, code_task)
                    st.subheader("💡 CodeCraft's Solution (Demo)")
                    st.markdown(response)
                    st.session_state.code_history.append({
                        'task': code_task,
                        'language': programming_language,
                        'question': code_prompt,
                        'response': response
                    })
                else:
                    llm = get_llm()
                    if not llm:
                        st.error("❌ API key issue. Check GROQ_API_KEY.")
                    else:
                        try:
                            code_system_prompt = f"""You are CodeCraft, a {programming_language} expert. Task: {code_task}
                            Provide clear, commented code and explanations."""
                            
                            code_prompt_template = ChatPromptTemplate.from_messages([
                                ("system", code_system_prompt),
                                ("human", "{question}"),
                            ])
                            
                            chain = code_prompt_template | llm | StrOutputParser()
                            response = chain.invoke({"question": code_prompt})
                            
                            st.subheader("💡 CodeCraft's Solution")
                            st.markdown(response)
                            
                            st.session_state.code_history.append({
                                'task': code_task,
                                'language': programming_language,
                                'question': code_prompt,
                                'response': response
                            })
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            st.rerun()
        else:
            st.warning("Please enter a coding question.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit, LangChain, and Groq</p>
    </div>
    """,
    unsafe_allow_html=True
)
