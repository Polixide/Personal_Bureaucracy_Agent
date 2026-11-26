"""Shared presentation for the student workspace and procedure stages."""
from html import escape

import streamlit as st

from src.core.procedure import USER_STEPS
from src.ui.components import field, request_view


STAGE_GROUPS = (
    {"DOCUMENT_COLLECTION"}, {"EIKAMET_PREPARATION", "EIKAMET_READY"}, {"PAYMENT", "FINGERPRINT"},
    {"IRO_REVIEW"}, {"SUBMISSION", "TRACKING"},
)

STAGE_GUIDE = {
    "DOCUMENT_COLLECTION": ("Get your documents ready", "Once your evidence is ready, you can prepare or review your online application."),
    "EIKAMET_PREPARATION": ("Prepare your online application", "Review your information here and complete the real application on e-İkamet."),
    "EIKAMET_READY": ("Review your online application", "Use the actual form to continue to payment and fingerprinting."),
    "PAYMENT": ("Review your payment evidence", "The next step is recording your fingerprint appointment outcome and GI number."),
    "FINGERPRINT": ("Record fingerprinting and your GI number", "Once those details are confirmed, you can prepare your university review package."),
    "IRO_REVIEW": ("Prepare your university review package", "Review the files and prepare a draft for the International Relations Office (IRO)."),
    "SUBMISSION": ("Record physical submission", "Record what you actually submitted, then track your case."),
    "TRACKING": ("Track your permit decision", "Record a decision once you receive it from the authority."),
    "COMPLETE": ("Your reported outcome", "Your update is saved in this case."),
}


def render_theme():
    st.html("""<style>
    .stMainBlockContainer { max-width: 78rem; }
    [data-testid="stAppDeployButton"] { display: none; }
    .st-key-case_hero, .st-key-welcome_hero {
        background: linear-gradient(118deg, #142d44, #17606a);
        border-radius: 1.125rem;
        padding: clamp(1rem, 3vw, 2.25rem);
    }
    .st-key-case_strip { background: #153b50; border-radius: 0.75rem; padding: 0.85rem 1.2rem; }
    .st-key-case_hero :is(h1,h2,p), .st-key-welcome_hero :is(h1,h2,p), .st-key-case_strip p { color: #f4fbfc; }
    .st-key-welcome_hero [data-testid="stCaptionContainer"] p { color: #d2eaf0; }
    @media (max-width: 600px) {
        .st-key-student_navigation [data-testid="stHorizontalBlock"] { gap: 0.45rem; }
        .st-key-student_navigation button { padding-inline: 0.5rem; }
    }
    .st-key-case_hero h1, .st-key-welcome_hero h1 {
        font-size: clamp(1.7rem, 4vw, 2.85rem);
        letter-spacing: -0.035em;
        line-height: 1.12;
    }
    .st-key-case_hero [role="progressbar"] { background: #346071; }
    .st-key-student_navigation [data-testid="stBaseButton-primary"] {
        background: #d7f2ee;
        color: #123e42;
        border-color: #90cbc5;
    }
    .pba-journey { list-style: none; padding: 0; margin: 0.35rem 0 1rem; }
    .pba-journey li {
        align-items: center; display: flex; gap: 0.65rem; padding: 0.5rem 0.5rem;
        margin: 0 0 0.2rem; font-size: 0.88rem; border-radius: 0.55rem;
    }
    .pba-journey li[aria-current="step"] { background: #d7f2ee; color: #123e42; font-weight: 650; }
    .pba-journey span {
        flex: 0 0 1.5rem; width: 1.5rem; height: 1.5rem; display: inline-grid;
        place-items: center; border: 1px solid #83a4ac; border-radius: 50%; font-size: 0.77rem;
    }
    .pba-journey .pba-done span { background: #236a60; border-color: #236a60; color: white; }
    .st-key-next_task { border-left: 3px solid #328980; }
    </style>""")


def render_case_header(summary, *, compact=False):
    title, following = STAGE_GUIDE.get(summary["stage"], STAGE_GUIDE["DOCUMENT_COLLECTION"])
    if compact:
        with st.container(key="case_strip"):
            st.markdown(f"**{summary['step_text']}**")
            if summary["progress_text"]:
                st.write(summary["progress_text"])
        return
    with st.container(key="case_hero"):
        st.markdown(f"**{summary['step_text']}**")
        st.title(title)
        st.write(following)
        if summary["progress_text"]:
            st.write(summary["progress_text"])
        if summary["progress_fraction"] is not None:
            st.progress(summary["progress_fraction"])


def render_stage_rail(state, summary):
    completed = set(field(state.get("procedure"), "completed_stages", []) or [])
    rows = []
    for index, (label, stages) in enumerate(zip(USER_STEPS, STAGE_GROUPS), start=1):
        current = index == summary["number"]
        done = stages <= completed
        attributes = ' aria-current="step"' if current else ' class="pba-done"' if done else ""
        mark = "✓" if done and not current else str(index)
        status = " (completed)" if done and not current else ""
        rows.append(f'<li{attributes}><span aria-hidden="true">{mark}</span>{escape(label + status)}</li>')
    st.sidebar.markdown("**Your preparation**")
    st.sidebar.html('<ol class="pba-journey" aria-label="Preparation stages">' + "".join(rows) + '</ol>')


def render_student_navigation(view, thread_key):
    main_views = (
        ("next", "Overview", ":material/space_dashboard:"),
        ("documents", "Documents", ":material/description:"),
        ("details", "Your details", ":material/person:"),
    )
    with st.container(key="student_navigation", horizontal=True):
        for destination, label, icon in main_views:
            st.button(label, icon=icon, type="primary" if destination == view else "secondary",
                      key=f"workspace_{destination}_{thread_key}", on_click=request_view,
                      args=(thread_key, destination))
    with st.sidebar:
        st.button("Open current step", icon=":material/arrow_forward:", key=f"step_page_{thread_key}",
                  on_click=request_view, args=(thread_key, "current_step"))
        st.button("Ask about this step", icon=":material/chat:", key=f"ask_page_{thread_key}",
                  on_click=request_view, args=(thread_key, "copilot"))
        st.button("Checks and warnings", icon=":material/fact_check:", key=f"checks_page_{thread_key}",
                  on_click=request_view, args=(thread_key, "checks"))
