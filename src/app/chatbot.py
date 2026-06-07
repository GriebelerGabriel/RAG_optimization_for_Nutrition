"""Streamlit chatbot interface for RAG Nutrition system."""
import hashlib
import logging
import sys
import time
import webbrowser
from pathlib import Path

import pandas as pd
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


def _friendly_llm_name(model_name: str) -> str:
    """Convert raw model name to a human-friendly label."""
    name = model_name.lower()
    if "gpt-3.5" in name or "gpt3.5" in name:
        return "GPT-3.5 Turbo"
    if "gpt-4" in name:
        return "GPT-4o"
    if "llama3.1" in name and "8b" in name:
        return "Llama 3.1 8B"
    if "llama" in name:
        return model_name
    return model_name


def main():
    st.title("🥗 Assistente de Nutrição")
    st.markdown(
        "Faça perguntas sobre nutrição e alimentação saudável. "
        "As respostas são baseadas em guias e documentos científicos."
    )
    
    # --- Precompute top 3 ranking for dropdown labels ---
    _top3: dict[str, str] = {}
    _results_dir = _PROJECT_ROOT / "results"
    _ragas_csv = _results_dir / "ragas_benchmark_results.csv"
    _det_csv = _results_dir / "deterministic_benchmark_results.csv"
    if _ragas_csv.exists() or _det_csv.exists():
        _ranking_rows = []
        if _ragas_csv.exists():
            _rdf = pd.read_csv(_ragas_csv)
            _ragas_metrics = [c for c in _rdf.columns if c not in ("config", "config_path", "avg_latency_ms")]
            for _, row in _rdf.iterrows():
                _ranking_rows.append({"config": row["config"], "ragas_mean": row[_ragas_metrics].mean()})
        if _det_csv.exists():
            _ddf = pd.read_csv(_det_csv)
            _det_metrics = [c for c in _ddf.columns if c not in ("config", "config_path", "avg_latency_ms")]
            _det_map = {row["config"]: row[_det_metrics].mean() for _, row in _ddf.iterrows()}
            for entry in _ranking_rows:
                entry["det_mean"] = _det_map.get(entry["config"], 0.0)
            _existing = {e["config"] for e in _ranking_rows}
            for name, score in _det_map.items():
                if name not in _existing:
                    _ranking_rows.append({"config": name, "ragas_mean": 0.0, "det_mean": score})
        if _ranking_rows:
            _rank_df = pd.DataFrame(_ranking_rows).fillna(0)
            _rank_df["composite"] = _rank_df.get("ragas_mean", 0) * 0.5 + _rank_df.get("det_mean", 0) * 0.5
            _rank_df = _rank_df.sort_values("composite", ascending=False).head(3)
            _medals = ["🥇", "🥈", "🥉"]
            for i, (_, row) in enumerate(_rank_df.iterrows()):
                _top3[row["config"]] = _medals[i] if i < len(_medals) else f"#{i+1}"

    # Widen sidebar via CSS
    st.markdown(
        "<style>"
        "section[data-testid='stSidebar'] { width: 380px !important; }"
        "section[data-testid='stSidebar'] .block-container { padding-top: 2rem; }"
        "</style>",
        unsafe_allow_html=True,
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuração")

        # Config file selection
        config_dir = Path("configs")
        config_files = sorted(config_dir.rglob("*.yaml")) if config_dir.exists() else []

        if config_files:
            config_options = {f.stem: str(f) for f in config_files}
            option_keys = sorted(config_options.keys())
            # Build display labels with medals
            _display_keys = []
            for k in option_keys:
                _cfg_name = k.replace("tune_", "").replace("fair_", "").replace("no_rag_", "no_rag_")
                _medal = _top3.get(k) or _top3.get(_cfg_name)
                _display_keys.append(f"{_medal} {k}" if _medal else k)
            default_key = "baseline" if "baseline" in option_keys else option_keys[0]
            selected_display = st.selectbox(
                "Configuração do RAG",
                options=_display_keys,
                index=option_keys.index(default_key),
                help=(
                    "🥇🥈🥉 indicam as 3 melhores configurações segundo o benchmark.\n\n"
                    "**Como o ranking funciona:**\n"
                    "- **RAGAS (50%)**: média de faithfulness, answer_relevancy, "
                    "context_precision, context_recall e answer_correctness — "
                    "avaliados por um LLM (Gemini 2.5 Flash) como juiz.\n"
                    "- **Métricas determinísticas (50%)**: média de BLEU, ROUGE-L, "
                    "METEOR, BERTScore F1 e Similaridade Semântica.\n\n"
                    "O score final é a média ponderada 50/50 dessas duas pontuações."
                ),
            )
            selected_config = option_keys[_display_keys.index(selected_display)]
            config_path = config_options[selected_config]
        else:
            st.warning("Nenhum arquivo de configuração encontrado em configs/")
            config_path = None

        # Compact config details — single expander, all in one table
        if config_path:
            try:
                config = Config.from_yaml(config_path)
                with st.expander("📋 Detalhes da config", expanded=False):
                    st.markdown(
                        f"| | Valor |\n|---|---|\n"
                        f"| **Chunker** | {config.chunker.type} |\n"
                        f"| **Chunk size** | {config.chunker.chunk_size} |\n"
                        f"| **Overlap** | {config.chunker.overlap} |\n"
                        f"| **Embedding** | {config.embedding.type} |\n"
                        f"| **Embed model** | `{config.embedding.model_name}` |\n"
                        f"| **Retriever** | {config.retriever.type} |\n"
                        f"| **Top K** | {config.retriever.top_k} |\n"
                        f"| **Score thresh** | {config.retriever.score_threshold} |\n"
                        f"| **Reranker** | {config.reranker.type} |\n"
                        + (f"| **Rerank model** | `{config.reranker.model_name}` |\n"
                           f"| **Rerank K** | {config.reranker.top_k} |\n"
                           if config.reranker.type != "none" else "")
                        + f"| **LLM** | `{_friendly_llm_name(config.llm.model_name)}` |\n"
                        f"| **Temperature** | {config.llm.temperature} |\n"
                        f"| **Seed** | {config.llm.seed} |\n"
                        f"| **Max tokens** | {config.llm.max_tokens} |"
                    )
            except Exception as e:
                st.error(f"Erro ao carregar config: {e}")

        # Benchmark report link
        _report_path = _results_dir / "benchmark_report.html"
        if _report_path.exists():
            if st.button("📊 Ver relatório benchmark"):
                webbrowser.open(f"file://{_report_path}")

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
