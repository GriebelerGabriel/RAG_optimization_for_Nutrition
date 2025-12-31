"""Streamlit chatbot interface for RAG Nutrition system."""
import hashlib
import logging
import sys
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.pipeline import RAGPipeline
from src.core.config import Config


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


# Page config
st.set_page_config(
    page_title="Assistente de Nutrição",
    page_icon="🥗",
    layout="centered",
)


@st.cache_resource
def load_pipeline(config_path: str, config_fingerprint: str) -> RAGPipeline:
    """Load and cache the RAG pipeline."""
    return RAGPipeline.from_config(config_path)


def _get_file_fingerprint(path: str) -> str:
    data = Path(path).read_bytes()
    return hashlib.md5(data).hexdigest()


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
            option_keys = sorted(config_options.keys())
            default_key = "baseline" if "baseline" in option_keys else option_keys[0]
            selected_config = st.selectbox(
                "Configuração do RAG",
                options=option_keys,
                index=option_keys.index(default_key),
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
            elif "pipeline_error" in st.session_state:
                st.error(f"Erro ao carregar pipeline: {st.session_state['pipeline_error']}")
            else:
                with st.spinner("Buscando informações..."):
                    try:
                        config_fingerprint = _get_file_fingerprint(config_path)
                        if st.session_state.get("pipeline_config_path") != config_path:
                            st.session_state.pop("pipeline", None)
                            st.session_state.pop("pipeline_error", None)
                            st.session_state["pipeline_config_path"] = config_path
                        pipeline = st.session_state.get("pipeline")
                        if pipeline is None:
                            t0 = time.perf_counter()
                            pipeline = load_pipeline(config_path, config_fingerprint)
                            st.session_state["pipeline"] = pipeline
                            logger.info(
                                "pipeline_loaded config=%s elapsed_ms=%.2f",
                                config_path,
                                (time.perf_counter() - t0) * 1000,
                            )

                        t_retrieve = time.perf_counter()
                        sources = pipeline.retrieve(prompt)
                        logger.info(
                            "retrieved query=%r k=%d elapsed_ms=%.2f top_sources=%s",
                            prompt,
                            len(sources),
                            (time.perf_counter() - t_retrieve) * 1000,
                            [
                                {
                                    "id": d.id,
                                    "score": float(d.score or 0.0),
                                    "source": d.metadata.get("filename", d.metadata.get("source", "unknown")),
                                }
                                for d in sources[:5]
                            ],
                        )

                        t_gen = time.perf_counter()
                        response = pipeline.generate(prompt, sources)
                        logger.info(
                            "generated elapsed_ms=%.2f response_chars=%d",
                            (time.perf_counter() - t_gen) * 1000,
                            len(response or ""),
                        )
                        
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
