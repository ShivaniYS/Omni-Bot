
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Omni-Bot - Your AI Companion",
    page_icon="🤖",
    layout="wide"
)

# Configuration - Streamlit Cloud compatible
# Use Streamlit secrets for API keys, fallback to environment variables for local development
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "Omni-Bot-Streamlit"))

# LangSmith Tracing Configuration (optional)
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Initialize session state
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
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = not bool(groq_api_key)

# Sidebar configuration
with st.sidebar:
    st.title("🤖 Omni-Bot Settings")
    
    # API Status
    st.subheader("🔑 API Status")
    if groq_api_key:
        st.success("✅ Groq API Key: Configured")
        st.session_state.demo_mode = False
    else:
        st.warning("🔶 Demo Mode - Using sample responses")
        st.info("💡 To enable full AI features, add GROQ_API_KEY to Streamlit secrets")
        st.session_state.demo_mode = True
    
    # Show LangSmith status
    langsmith_status = "✅ Configured" if langchain_api_key else "❌ Not Configured"
    st.write(f"LangSmith Tracing: {langsmith_status}")
    
    # Model Selection
    st.subheader("🧠 Model Settings")
    mode = st.selectbox(
        "Choose Your Assistant Mode",
        ["Brainy Buddy", "DocuMind", "CodeCraft"],
        help="Select what type of assistance you need"
    )
    
    # Update current mode
    st.session_state.current_mode = mode
    
    # Model parameters (only show if API key is available)
    if groq_api_key:
        engine = st.selectbox(
            "Select Groq Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 2000, 512, 50)
    else:
        # Default values for demo mode
        engine = "llama-3.1-8b-instant"
        temperature = 0.7
        max_tokens = 512
        
        st.info("""
        **Available in Full Mode:**
        - Model selection
        - Temperature control
        - Token limits
        
        Add GROQ_API_KEY to unlock!
        """)
    
    # Session management for document mode
    if mode == "DocuMind":
        session_id = st.text_input("Session ID for document chat", value="default_session")
    
    # Clear history buttons
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
            st.session_state.rag_chain = None
            st.session_state.processed_files = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About Omni-Bot")
    st.markdown("""
    Your all-in-one AI assistant that can:
    - 💬 Have intelligent conversations (Brainy Buddy)
    - 📚 Answer questions from uploaded documents (DocuMind)  
    - 💻 Help with coding tasks (CodeCraft)
    """)

# Main app title
st.title("🤖 Omni-Bot - Your Intelligent AI Companion")

# Show mode banner
if st.session_state.demo_mode:
    st.warning("🔶 **Demo Mode** - Add GROQ_API_KEY to Streamlit secrets for full AI capabilities")
else:
    st.success("🚀 **Full AI Mode** - All features enabled!")

st.markdown("---")

# Demo responses for when no API key is provided
def get_demo_response(mode, prompt, programming_language="Python", code_task="Write Code"):
    """Provide demo responses when no API key is available"""
    if mode == "Brainy Buddy":
        demo_responses = [
            f"Hello! I'm Brainy Buddy in demo mode. You asked: '{prompt}'. In the full version with a Groq API key, I'd provide a detailed and intelligent response to your question.",
            f"That's an interesting question about '{prompt}'. With a Groq API key, I'd be able to give you a comprehensive answer with real-time AI capabilities.",
            f"Thanks for your message! To get actual AI-powered responses, please add GROQ_API_KEY to Streamlit secrets. Demo response to: '{prompt}'"
        ]
        return demo_responses[len(prompt) % 3]
    
    elif mode == "CodeCraft":
        demo_code_responses = {
            "Write Code": f"""# Demo Response for {programming_language}
            
In demo mode, I can show you what CodeCraft would do:

For task: {prompt}

With a Groq API key, I would:
1. Write clean, efficient {programming_language} code
2. Add proper comments and documentation
3. Explain the implementation approach
4. Provide usage examples

Get your free API key from https://console.groq.com and add GROQ_API_KEY to Streamlit secrets to unlock full functionality!""",

            "Debug/Explain": f"""# Debugging Help - Demo Mode

For: {prompt}

With Groq API, I would:
- Analyze your code for errors
- Explain what's happening
- Suggest fixes
- Provide corrected code

Add GROQ_API_KEY to Streamlit secrets for real debugging assistance!""",

            "Optimize": f"""# Code Optimization - Demo Mode

Task: {prompt}

With API access, I would:
- Identify performance bottlenecks
- Suggest optimizations
- Provide benchmark comparisons
- Show best practices

Enable full features by adding GROQ_API_KEY to Streamlit secrets!"""
        }
        return demo_code_responses.get(code_task, demo_code_responses["Write Code"])
    
    elif mode == "DocuMind":
        return f"""📚 DocuMind Demo Response

Question: {prompt}

In full mode with Groq API, I would:
- Search through your uploaded documents
- Find relevant information
- Provide accurate answers with citations
- Maintain conversation context

Add GROQ_API_KEY to Streamlit secrets to process documents and get real answers!"""

# Initialize LLM if API key is provided
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
        return None  # Demo mode

# BRAINY BUDDY MODE - Conversational Q&A
if st.session_state.current_mode == "Brainy Buddy":
    st.header("💬 Brainy Buddy - Intelligent Conversations")
    
    if st.session_state.demo_mode:
        st.info("🔶 Demo Mode - Add GROQ_API_KEY to Streamlit secrets for real AI responses")
    else:
        st.markdown("Have meaningful conversations with your AI companion")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        llm = get_llm()
        
        # Add user message to chat
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if not llm:
                # Demo mode
                with st.spinner("💭 Demo mode..."):
                    response = get_demo_response("Brainy Buddy", prompt)
                    st.markdown(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
            else:
                # Real AI mode
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Create conversation context
                        messages = [
                            ("system", "You are Brainy Buddy, a helpful, friendly, and intelligent AI assistant. Have engaging conversations and provide useful information. Keep responses concise but informative.")
                        ]
                        
                        # Add conversation history (last 6 messages for context)
                        for msg in st.session_state.conversation_history[-6:]:
                            role = "human" if msg["role"] == "user" else "assistant"
                            messages.append((role, msg["content"]))
                        
                        prompt_template = ChatPromptTemplate.from_messages(messages)
                        chain = prompt_template | llm | StrOutputParser()
                        response = chain.invoke({"question": prompt})
                        
                        st.markdown(response)
                        st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

# DOCUMENT MODE - PDF Q&A
elif st.session_state.current_mode == "DocuMind":
    st.header("📚 DocuMind - Document Intelligence")
    
    if st.session_state.demo_mode:
        st.info("🔶 Demo Mode - Add GROQ_API_KEY to Streamlit secrets to process real documents")
        st.markdown("In demo mode, you can upload documents but will need an API key for actual Q&A.")
    else:
        st.markdown("Upload documents and ask questions about their content")
    
    # File upload (available in both modes)
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    if uploaded_files and not st.session_state.demo_mode:
        # Only process documents if we have API key
        llm = get_llm()
        if not llm:
            st.error("❌ API key issue. Please check your GROQ_API_KEY in Streamlit secrets.")
        else:
            # Show currently processed files
            if st.session_state.processed_files:
                st.info(f"📁 Processed files: {', '.join(st.session_state.processed_files)}")
            
            if st.session_state.vectorstore is None or st.button("Reprocess Documents"):
                # Process uploaded files
                with st.spinner("📄 Processing documents... This may take a moment."):
                    documents = []
                    temp_files = []
                    
                    try:
                        for uploaded_file in uploaded_files:
                            # Create temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                                temp_file.write(uploaded_file.getbuffer())
                                temp_files.append(temp_file.name)
                            
                            # Load PDF
                            loader = PyPDFLoader(temp_file.name)
                            docs = loader.load()
                            # Add source information
                            for doc in docs:
                                doc.metadata['source'] = uploaded_file.name
                            documents.extend(docs)
                        
                        if documents:
                            # Split documents
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, 
                                chunk_overlap=200
                            )
                            splits = text_splitter.split_documents(documents)
                            
                            # Create vector store
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                            vectorstore = Chroma.from_documents(
                                documents=splits, 
                                embedding=embeddings
                            )
                            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                            
                            # Create RAG chain
                            contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context from the chat history, formulate a standalone question which can be understood without the chat history. Return the standalone question."""
                            
                            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                                ("system", contextualize_q_system_prompt),
                                MessagesPlaceholder(variable_name="chat_history"),
                                ("human", "{input}"),
                            ])
                            
                            history_aware_retriever = create_history_aware_retriever(
                                llm, retriever, contextualize_q_prompt
                            )
                            
                            # QA prompt
                            qa_system_prompt = """You are DocuMind, an expert at analyzing documents and providing accurate answers based on the provided context. Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, say so. Don't make up information.
                            
                            Context: {context}"""
                            
                            qa_prompt = ChatPromptTemplate.from_messages([
                                ("system", qa_system_prompt),
                                MessagesPlaceholder(variable_name="chat_history"),
                                ("human", "{input}"),
                            ])
                            
                            question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)
                            rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)
                            
                            # Store in session state
                            st.session_state.vectorstore = vectorstore
                            st.session_state.rag_chain = rag_chain
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            
                            st.success(f"✅ Processed {len(uploaded_files)} document(s) with {len(splits)} chunks! You can now ask questions.")
                        
                        # Cleanup temp files
                        for temp_file in temp_files:
                            try:
                                os.unlink(temp_file)
                            except:
                                pass
                                
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    # Question input
    user_question = st.text_input("Enter your question about the documents:", key="doc_question")
    
    if user_question:
        if st.session_state.demo_mode:
            # Demo mode response
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                response = get_demo_response("DocuMind", user_question)
                st.markdown(response)
        elif st.session_state.rag_chain is not None:
            # Real document Q&A
            with st.spinner("🔍 Searching documents..."):
                try:
                    # Session history management
                    def get_session_history(session_id: str) -> BaseChatMessageHistory:
                        if session_id not in st.session_state.document_store:
                            st.session_state.document_store[session_id] = ChatMessageHistory()
                        return st.session_state.document_store[session_id]
                    
                    conversational_rag_chain = RunnableWithMessageHistory(
                        st.session_state.rag_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )
                    
                    response = conversational_rag_chain.invoke(
                        {"input": user_question},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    st.subheader("📝 Answer:")
                    st.write(response['answer'])
                    
                    # Show chat history
                    with st.expander("View Conversation History"):
                        session_history = get_session_history(session_id)
                        if session_history.messages:
                            for message in session_history.messages:
                                icon = "👤" if message.type == "human" else "🤖"
                                st.write(f"{icon} **{message.type}:** {message.content}")
                                st.markdown("---")
                        else:
                            st.write("No conversation history yet.")
                            
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
        else:
            st.warning("Please upload and process documents first, or add GROQ_API_KEY to Streamlit secrets for demo mode.")

# CODE ASSISTANT MODE
elif st.session_state.current_mode == "CodeCraft":
    st.header("💻 CodeCraft - Your Coding Assistant")
    
    if st.session_state.demo_mode:
        st.info("🔶 Demo Mode - Add GROQ_API_KEY to Streamlit secrets for real code assistance")
    else:
        st.markdown("Get help with programming, code explanation, and debugging")
    
    # Code-specific parameters
    col1, col2 = st.columns(2)
    with col1:
        programming_language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "SQL", "TypeScript", "HTML/CSS", "Other"]
        )
    with col2:
        code_task = st.selectbox(
            "Type of Assistance",
            ["Write Code", "Debug/Explain", "Optimize", "Learn Concepts", "Code Review"]
        )
    
    # Display code history
    if st.session_state.code_history:
        st.subheader("📚 Recent Code Interactions")
        for i, interaction in enumerate(reversed(st.session_state.code_history[-3:])):
            with st.expander(f"💻 {interaction['task']} - {interaction['language']} (Interaction {len(st.session_state.code_history)-i})"):
                st.markdown("**Your Question/Code:**")
                st.code(interaction['question'], language=interaction.get('language', 'text').lower())
                st.markdown("**CodeCraft's Response:**")
                st.markdown(interaction['response'])
    
    # Code input
    code_prompt = st.text_area(
        "Describe your coding task or paste your code:",
        height=150,
        placeholder="e.g., Write a Python function to calculate factorial, or paste code you need help with..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Get Code Help 🚀", type="primary", use_container_width=True):
            if code_prompt:
                with st.spinner("💻 Working on your code..."):
                    if st.session_state.demo_mode:
                        # Demo mode response
                        response = get_demo_response("CodeCraft", code_prompt, programming_language, code_task)
                        
                        st.subheader("💡 CodeCraft's Solution (Demo)")
                        st.markdown(response)
                        
                        # Save to history
                        st.session_state.code_history.append({
                            'task': code_task,
                            'language': programming_language,
                            'question': code_prompt,
                            'response': response
                        })
                    else:
                        # Real AI mode
                        llm = get_llm()
                        if not llm:
                            st.error("❌ API key issue. Please check your GROQ_API_KEY in Streamlit secrets.")
                        else:
                            try:
                                # Code-specific prompt
                                code_system_prompt = f"""You are CodeCraft, an expert programming assistant specializing in {programming_language}.
                                
                                Task type: {code_task}
                                
                                Please provide:
                                - Clear, well-commented code when writing or optimizing
                                - Detailed explanations when debugging or explaining
                                - Best practices and potential improvements
                                - Error explanations and solutions when debugging
                                - Be concise but thorough in your responses.
                                - Format code properly with syntax highlighting.
                                """
                                
                                code_prompt_template = ChatPromptTemplate.from_messages([
                                    ("system", code_system_prompt),
                                    ("human", "{question}"),
                                ])
                                
                                chain = code_prompt_template | llm | StrOutputParser()
                                response = chain.invoke({"question": code_prompt})
                                
                                # Display response
                                st.subheader("💡 CodeCraft's Solution")
                                st.markdown(response)
                                
                                # Save to history
                                st.session_state.code_history.append({
                                    'task': code_task,
                                    'language': programming_language,
                                    'question': code_prompt,
                                    'response': response
                                })
                                
                            except Exception as e:
                                st.error(f"Error generating code response: {str(e)}")
                
                st.rerun()
            else:
                st.warning("Please enter a coding question or paste some code.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, LangChain, and Groq | LangSmith Tracing: <strong>{}</strong> | Mode: <strong>{}</strong></p>
    </div>
    """.format(
        "Enabled" if langchain_api_key else "Disabled",
        "Demo" if st.session_state.demo_mode else "Full AI"
    ),
    unsafe_allow_html=True
)

# Instructions for setup (only show in demo mode)
if st.session_state.demo_mode:
    with st.expander("🔧 Setup Instructions for Full Features"):
        st.markdown("""
        ### To enable full AI capabilities:
        
        1. **Get a free Groq API key:**
           - Visit [https://console.groq.com](https://console.groq.com)
           - Sign up for a free account
           - Generate an API key
        
        2. **Add to Streamlit Cloud Secrets:**
           - Go to your app settings in Streamlit Cloud
           - Under "Secrets", add:
           ```toml
           GROQ_API_KEY = "your_actual_api_key_here"
           LANGCHAIN_API_KEY = "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"
           LANGCHAIN_PROJECT = "Omni-Bot-Streamlit"
           ```
        
        3. **Redeploy the app**
        
        That's it! You'll get access to:
        - Real AI conversations
        - Document processing and Q&A
        - Code generation and debugging
        - Model customization
        """)