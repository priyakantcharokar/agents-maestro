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
    QUICK_FOLLOWUPS = [
    "Could you outline Priyakant's most recent AI project?",
    "What certification does Priyakant value the most and why?",
    "Which industry challenge excites Priyakant right now?",
    "How does he keep his team motivated during large programmes?",
    "What books has Priyakant been reading lately?"
    ]

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False

        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self._handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True

        # Base response
        reply = response.choices[0].message.content

        # Append follow-up suggestions as HTML
        followup_html = "<hr style='border:0;border-top:1px solid #ccc;margin:12px 0;'>"
        followup_html += "<strong>You could also ask:</strong><ul style='margin-top:4px;'>"
        for q in self.QUICK_FOLLOWUPS:
            followup_html += f"<li>{q}</li>"
        followup_html += "</ul>"

        return reply + followup_html

# -------------------------------------------------------------------
# 6. Build Gradio interface  (exposed as `demo`)
# -------------------------------------------------------------------
me = Me()

demo = gr.ChatInterface(
    fn=me.chat,
    type="messages",
    chatbot=gr.Chatbot(
        label="Ask Priyakant",
        # avatar_images=("🧑‍💼", "🤖"),
        avatar_images=("user.png", "bot.png"),
        show_copy_button=True,
        type="messages"
    ),
    title="Know more about Priyakant, professionally 😉",
    description="<div style='text-align: center;'>🤝 I'm Priyakant's digital assistant. Ask anything about his work, skills, or how he can help you!</div>",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="gray",
        font=["Inter", "sans-serif"],
        neutral_hue="slate",
    ),
    examples = [
    ["What leadership roles has Priyakant held in large-scale digital transformation programs?"],
    ["Can you summarize Priyakant's experience with cloud architecture and Generative AI?"],
    ["What certifications or technical credentials does Priyakant hold?"],
    ["How does Priyakant's background align with senior technology leadership roles?"],
    ["Can you share highlights from his Senior Management Program at IIM Calcutta?"],
    ["What industries has Priyakant delivered strategic architecture solutions for?"],
    ["Does Priyakant publish thought leadership articles or technical blogs?"],
    ["How does Priyakant mentor teams or contribute to capability building?"],
    ["What is Priyakant's approach to cross-functional stakeholder communication?"],
    ["How can I connect with Priyakant to explore a leadership opportunity?"]
],
)

# -------------------------------------------------------------------
# 7. Local dev launcher  (HF Spaces auto-launches `demo`)
# -------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
