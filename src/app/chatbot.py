"""Streamlit chatbot interface for RAG Nutrition system."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.core.pipeline import RAGPipeline
from src.core.config import Config


# Page config
st.set_page_config(
    page_title="Assistente de Nutrição",
    page_icon="🥗",
    layout="centered",
)


@st.cache_resource
def load_pipeline(config_path: str) -> RAGPipeline:
    """Load and cache the RAG pipeline."""
    return RAGPipeline.from_config(config_path)


def main():
    st.title("🥗 Assistente de Nutrição")
    st.markdown(
        "Faça perguntas sobre nutrição e alimentação saudável. "
        "As respostas são baseadas em guias e documentos científicos."
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuração")
        
        # Config file selection
        config_dir = Path("configs")
        config_files = list(config_dir.glob("*.yaml")) if config_dir.exists() else []
        
        if config_files:
            config_options = {f.stem: str(f) for f in config_files}
            selected_config = st.selectbox(
                "Configuração do RAG",
                options=list(config_options.keys()),
                index=0,
            )
            config_path = config_options[selected_config]
        else:
            st.warning("Nenhum arquivo de configuração encontrado em configs/")
            config_path = None
        
        st.divider()
        
        # Show current config details
        if config_path:
            try:
                config = Config.from_yaml(config_path)
                st.markdown("**Configuração atual:**")
                st.markdown(f"- Chunker: `{config.chunker.type}`")
                st.markdown(f"- Embedding: `{config.embedding.type}`")
                st.markdown(f"- Retriever: `{config.retriever.type}`")
                st.markdown(f"- Reranker: `{config.reranker.type}`")
                st.markdown(f"- LLM: `{config.llm.type}`")
            except Exception as e:
                st.error(f"Erro ao carregar config: {e}")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Limpar conversa"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Fontes consultadas"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Faça sua pergunta sobre nutrição..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if config_path is None:
                st.error("Configure um arquivo de configuração primeiro.")
            else:
                with st.spinner("Buscando informações..."):
                    try:
                        pipeline = load_pipeline(config_path)
                        response, sources = pipeline.query(prompt)
                        
                        st.markdown(response)
                        
                        # Show sources
                        source_texts = []
                        if sources:
                            with st.expander("📚 Fontes consultadas"):
                                for i, doc in enumerate(sources):
                                    source_name = doc.metadata.get("filename", doc.metadata.get("source", f"Fonte {i+1}"))
                                    score = doc.score or 0
                                    st.markdown(f"**{source_name}** (relevância: {score:.2f})")
                                    st.markdown(f"> {doc.content[:200]}...")
                                    source_texts.append(f"{source_name} ({score:.2f})")
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": source_texts,
                        })
                    
                    except Exception as e:
                        st.error(f"Erro ao processar pergunta: {e}")
                        st.exception(e)


if __name__ == "__main__":
    main()
