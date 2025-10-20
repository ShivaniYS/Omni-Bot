# OmniBot - Your AI Companion

A multitasking chatbot built with Streamlit and LangChain that provides:
- 💬 Intelligent conversations (Brainy Buddy)
- 📚 Document Q&A (DocuMind)
- 💻 Coding assistance (CodeCraft)

## Setup for Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys
4. Run: `streamlit run app.py`

## Deployment

This app is deployed on Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set up your secrets in the Streamlit Cloud dashboard
5. Deploy!

## Environment Variables

Create a `.env` file with:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=your_project_name_here

