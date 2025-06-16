# app.py  – put this in the root of your Space repo
import json, os, requests
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# -------------------------------------------------------------------
# 1. ENV ­– works locally (.env) and on HF Spaces (Secrets tab)
# -------------------------------------------------------------------
load_dotenv(override=True)          # noop on Spaces unless you add a .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------------------------
# 2. PUSHOVER helper  (optional – remove if unused)
# -------------------------------------------------------------------
def push(text: str):
    """Send a quick push notification (Pushover)."""
    if os.getenv("PUSHOVER_TOKEN") and os.getenv("PUSHOVER_USER"):
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            },
            timeout=10,
        )

# -------------------------------------------------------------------
# 3. Tool functions + JSON specs
# -------------------------------------------------------------------
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} ({email}) • {notes}")
    return {"status": "saved"}

def record_unknown_question(question):
    push(f"Unknown question logged: {question}")
    return {"status": "logged"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Store user-supplied contact info",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "User's email address"},
            "name":  {"type": "string", "description": "User's name (if given)"},
            "notes": {"type": "string", "description": "Extra context"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Log any question the assistant couldn't answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unanswered question"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

# -------------------------------------------------------------------
# 4. Helpers
# -------------------------------------------------------------------
def pdf_to_text(pdf_path: Path) -> str:
    """Concatenate all pages of a PDF into one string."""
    reader = PdfReader(str(pdf_path))
    return "".join(page.extract_text() or "" for page in reader.pages)

# -------------------------------------------------------------------
# 5. Main class
# -------------------------------------------------------------------
class Me:
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.name   = "Priyakant Charokar"

        base = Path(__file__).parent / "me"          # me/ folder next to app.py
        self.linkedin = pdf_to_text(base / "linkedin.pdf")
        self.resume   = pdf_to_text(base / "PriyakantCharokar.pdf")
        self.summary  = (base / "summary.txt").read_text(encoding="utf-8")

    # -- build prompt ------------------------------------------------
    def system_prompt(self) -> str:
        return (
            f"You are acting as {self.name}. Answer questions about his career, "
            f"skills, and experience professionally and engagingly.\n\n"
            f"## Summary:\n{self.summary}\n\n"
            f"## LinkedIn Profile:\n{self.linkedin}\n\n"
            f"## Resume:\n{self.resume}\n\n"
            "If unsure of an answer, call record_unknown_question. "
            "If the user seems interested in contact, ask for their email and call record_user_details."
        )

    # -- tool-call handler -------------------------------------------
    def _handle_tool_calls(self, tool_calls):
        responses = []
        for tc in tool_calls:
            fn = globals().get(tc.function.name)
            args = json.loads(tc.function.arguments)
            out  = fn(**args) if fn else {}
            responses.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)}
            )
        return responses

    # -- chat function (Gradio expects signature: message, history) --
    def chat(self, message, history):
        msgs = (
            [{"role": "system", "content": self.system_prompt()}]
            + history
            + [{"role": "user", "content": message}]
        )

        while True:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
                tools=tools,
            )
            if resp.choices[0].finish_reason == "tool_calls":
                assistant_msg = resp.choices[0].message
                msgs.append(assistant_msg)
                msgs.extend(self._handle_tool_calls(assistant_msg.tool_calls))
            else:
                return resp.choices[0].message.content

# -------------------------------------------------------------------
# 6. Build Gradio interface  (exposed as `demo`)
# -------------------------------------------------------------------
me = Me()

demo = gr.ChatInterface(
    fn=me.chat,
    type="messages",
    chatbot=gr.Chatbot(
        label="Ask Priyakant 🤖",
        avatar_images=("🧑‍💼", "🤖"),
        show_copy_button=True,
        type="messages",
    ),
    title="Ask Priyakant Charokar",
    description="🤝 I'm Priyakant's digital assistant. Ask anything about his work, skills, or how he can help you!",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="gray",
        font=["Inter", "sans-serif"],
    ),
    examples=[
        ["What industries has Priyakant worked with as a lead architect?"],
        ["How has Priyakant applied Generative AI in enterprise projects?"],
        ["Describe his approach to intelligent data platforms."],
        ["What leadership roles has he held in digital-transformation programs?"],
        ["How can I collaborate with him on a cloud-native initiative?"],
    ],
)

# -------------------------------------------------------------------
# 7. Local dev launcher  (HF Spaces auto-launches `demo`)
# -------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
