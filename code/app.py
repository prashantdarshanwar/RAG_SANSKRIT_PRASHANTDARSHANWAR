import streamlit as st
import os
import tempfile

from src.pipeline.rag_pipeline import SanskritRAGPipeline

st.set_page_config(
    page_title="Sanskrit RAG Assistant",
    page_icon="📜",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&family=Cinzel:wght@400;600&display=swap');

:root {
    --bg-deep:       #0b0a08;
    --bg-card:       #13120f;
    --bg-raised:     #1c1a15;
    --border:        #2e2b22;
    --border-glow:   #8b6914;
    --gold:          #c9972b;
    --gold-light:    #e8c06a;
    --gold-pale:     #f5e4b0;
    --text-primary:  #ede8d8;
    --text-muted:    #8a8170;
    --text-dim:      #5a5545;
    --accent-red:    #8b3a2a;
    --accent-teal:   #2a6b5e;
    --success:       #3d7a5c;
    --warning:       #7a5e2a;
    --error:         #7a2a2a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 2px; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 50% 0%, rgba(201,151,43,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.35em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Cinzel', serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 600;
    color: var(--gold-pale);
    line-height: 1.1;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
    text-shadow: 0 0 60px rgba(201,151,43,0.25);
}
.hero-sub {
    font-size: 1.15rem;
    font-style: italic;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-ornament {
    margin: 1.2rem auto 0;
    color: var(--gold);
    font-size: 1.1rem;
    letter-spacing: 0.5em;
    opacity: 0.6;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--border-glow) 0%, transparent 100%);
    opacity: 0.4;
}

/* ── Cards / panels ── */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0.35;
}

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 1px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-ready   { color: #6dbf91; border-color: #3d7a5c44; background: #1a3d2a; }
.badge-waiting { color: #c9972b; border-color: #8b691444; background: #2e230a; }
.badge-error   { color: #e08070; border-color: #7a2a2a44; background: #2e1010; }

/* ── Streamlit widget overrides ── */

/* file uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1px dashed var(--border-glow) !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-muted) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1rem !important;
}

/* text input */
[data-testid="stTextInput"] input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text-primary) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,151,43,0.12) !important;
}
[data-testid="stTextInput"] label {
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}

/* slider */
[data-testid="stSlider"] label {
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: var(--gold) !important;
    border: 2px solid var(--bg-deep) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }

/* buttons */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold-light) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 1px !important;
    transition: all 0.22s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(201,151,43,0.1) !important;
    border-color: var(--gold-light) !important;
    color: var(--gold-pale) !important;
    box-shadow: 0 0 20px rgba(201,151,43,0.15) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* alerts */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1rem !important;
    border-left: 3px solid !important;
}
.stSuccess { border-color: var(--success) !important; background: #0f2018 !important; }
.stWarning { border-color: var(--warning) !important; background: #1e180a !important; }
.stError   { border-color: var(--error)   !important; background: #1e0f0f !important; }

/* expander */
[data-testid="stExpander"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* spinner */
[data-testid="stSpinner"] { color: var(--gold) !important; }
[data-testid="stSpinner"] > div > div { border-top-color: var(--gold) !important; }

/* response box */
.response-box {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 2px;
    padding: 1.5rem 1.8rem;
    font-size: 1.12rem;
    line-height: 1.85;
    color: var(--text-primary);
    margin-top: 0.5rem;
}

/* chunk card */
.chunk-card {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.9rem;
    position: relative;
}
.chunk-index {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* footer */
.footer {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    color: var(--text-dim);
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

/* divider */
.gold-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
    position: relative;
}
.gold-divider::before {
    content: '✦';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: var(--bg-deep);
    color: var(--gold);
    padding: 0 0.8rem;
    font-size: 0.7rem;
    opacity: 0.5;
}
/* Q&A response card */
.qa-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.qa-question-label, .qa-answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    padding: 0.55rem 1.5rem 0.3rem;
}
.qa-question-label { color: var(--text-dim); background: var(--bg-raised); }
.qa-answer-label   { color: var(--gold);     background: transparent; }
.qa-question {
    font-size: 1.08rem;
    color: var(--text-primary);
    padding: 0.4rem 1.5rem 1rem;
    background: var(--bg-raised);
    line-height: 1.6;
    border-bottom: none;
}
.qa-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--gold) 0%, transparent 80%);
    opacity: 0.35;
}
.qa-answer {
    font-size: 1.1rem;
    color: var(--text-primary);
    padding: 0.8rem 1.5rem 1.5rem;
    line-height: 1.85;
}

/* reference question cards */
.ref-question {
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
}
.ref-question:last-child { border-bottom: none; }
.ref-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    color: var(--gold);
    margin-right: 0.6rem;
}
.ref-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    padding: 0.1rem 0.45rem;
    border-radius: 1px;
}
.ref-sanskrit {
    font-size: 1.05rem;
    color: var(--text-primary);
    margin: 0.4rem 0 0.2rem;
    line-height: 1.5;
}
.ref-english {
    font-size: 0.85rem;
    font-style: italic;
    color: var(--text-muted);
    line-height: 1.4;
}
</style>

<div class="hero">
    <div class="hero-eyebrow">Retrieval-Augmented Generation</div>
    <div class="hero-title">Sanskrit Scholar</div>
    <div class="hero-sub">Illuminate ancient texts through the lens of modern AI</div>
    <div class="hero-ornament">❧ ✦ ❧</div>
</div>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "build_error" not in st.session_state:
    st.session_state.build_error = None


# ── Layout columns ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Upload & Build
# ═══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="section-label">01 &nbsp; Document Ingestion</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a Sanskrit PDF or TXT document",
        type=["pdf", "txt"],
        label_visibility="visible"
    )

    if uploaded_file is not None:
        # Auto-build whenever a new (or different) file is uploaded
        if uploaded_file.name != st.session_state.last_uploaded_name:
            st.session_state.pipeline_ready = False
            st.session_state.pipeline       = None
            st.session_state.build_error    = None
            st.session_state.last_uploaded_name = uploaded_file.name

            with st.spinner(f"Indexing **{uploaded_file.name}** — please wait…"):
                try:
                    temp_dir  = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    rag = SanskritRAGPipeline()
                    rag.build_pipeline(temp_path)

                    st.session_state.pipeline       = rag
                    st.session_state.pipeline_ready = True

                except Exception as e:
                    st.session_state.build_error = str(e)

        if st.session_state.pipeline_ready:
            st.success(f"**{uploaded_file.name}** indexed — ready to query.")
        elif st.session_state.build_error:
            st.error(f"Build failed: {st.session_state.build_error}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Status indicator
    st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)
    if st.session_state.pipeline_ready:
        st.markdown('<span class="badge badge-ready">● Active &amp; Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-waiting">○ Awaiting Document</span>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Retrieval settings ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">02 &nbsp; Retrieval Settings</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    top_k = st.slider(
        "Top-K Retrieved Passages",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ── Reference questions ────────────────────────────────────────────────────────
REFERENCE_QUESTIONS = [
    {
        "sanskrit": "अन्तिमे श्लोके लेखकः किम् उपदेशं ददाति ?",
        "english":  "What moral lesson does the author give in the final verse of \"The Foolish Servant\"?",
        "tag":      "The Foolish Servant",
    },
    {
        "sanskrit": "कालीदासः नवकवये कीदृशं उपायं अयोजयत् येन कविः लक्षरुप्यकाणि प्राप्नोत् ?",
        "english":  "What clever strategy did Kalidasa devise so that the poet could win one lakh rupees?",
        "tag":      "Kalidasa Story",
    },
    {
        "sanskrit": "वृद्धा वानरेभ्यः फलानि दत्त्वा घण्टां कथम् आनीतवती ? तस्याः चातुर्यं किम् ?",
        "english":  "How did the old woman bring the bell by giving fruits to the monkeys? What was her cleverness?",
        "tag":      "The Old Woman",
    },
    {
        "sanskrit": "देवः स्वर्गे भक्तम् किम् अवदत् ? तस्मात् वयं किम् शिक्षामः ?",
        "english":  "What did God say to the devotee in heaven? What do we learn from that?",
        "tag":      "The Devotee",
    },
    {
        "sanskrit": "अन्तिमे श्लोके कानि षट् गुणानि उक्तानि ? तेषां नामानि लिखत ।",
        "english":  "What are the six qualities mentioned in the final verse of the devotee story? Write their names.",
        "tag":      "Six Qualities",
    },
]

if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Query & Results
# ═══════════════════════════════════════════════════════════════════════════════
with right:
    # ── Reference questions ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">03 &nbsp; Reference Questions</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:var(--text-muted);'
        'letter-spacing:0.1em;margin-bottom:1rem;">Click any question to load it into the query box</p>',
        unsafe_allow_html=True
    )

    for i, q in enumerate(REFERENCE_QUESTIONS, start=1):
        col_q, col_btn = st.columns([5, 1], gap="small")
        with col_q:
            st.markdown(
                f'<div class="ref-question">'
                f'<span class="ref-num">Q{i}</span>'
                f'<span class="ref-tag">{q["tag"]}</span>'
                f'<div class="ref-sanskrit">{q["sanskrit"]}</div>'
                f'<div class="ref-english">{q["english"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_btn:
            if st.button("↗", key=f"refq_{i}", help="Load this question"):
                st.session_state.prefill_query = q["sanskrit"]
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Query input ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">04 &nbsp; Query the Text</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    query = st.text_input(
        "Sanskrit or English Query",
        value=st.session_state.prefill_query,
        label_visibility="visible"
    )

    search_clicked = st.button("✦  Generate Answer")

    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked:
        if not st.session_state.pipeline_ready:
            st.warning("Build the pipeline first (upload a document on the left).")
        elif not query.strip():
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving passages & composing answer…"):
                try:
                    result = st.session_state.pipeline.ask_question(
                        query=query,
                        top_k=top_k
                    )

                    # ── Q&A card ───────────────────────────────────────────────
                    st.markdown('<div class="section-label">Response</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="qa-card">'
                        f'<div class="qa-question-label">&#x092A;&#x094D;&#x0930;&#x0936;&#x094D;&#x0928;&#x0903; &middot; Question</div>'
                        f'<div class="qa-question">{query}</div>'
                        f'<div class="qa-divider"></div>'
                        f'<div class="qa-answer-label">&#x0909;&#x0924;&#x094D;&#x0924;&#x0930;&#x092E;&#x094D; &middot; Answer</div>'
                        f'<div class="qa-answer">{result["response"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Error: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Sanskrit Scholar &nbsp;·&nbsp; FAISS · Sentence Transformers · RAG · Streamlit</div>',
    unsafe_allow_html=True
)