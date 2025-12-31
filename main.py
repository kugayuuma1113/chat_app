from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
from sqlmodel import Field, SQLModel, create_engine, Session, select
from sqlalchemy import desc

# --- データベース設定（変更なし） ---
class ChatMessage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    role: str
    content: str

sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)
SQLModel.metadata.create_all(engine)

# --- FastAPI & AI設定 ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")  # テンプレートの場所を指定

MODEL_PATH = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

# 1. ブラウザでアクセスした時に最初の画面(index.html)を返す
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. チャット送信時の処理
@app.post("/chat", response_class=HTMLResponse)
def chat_endpoint(request: Request, prompt: str = Form(...)):
    """
    チャットメッセージを受け取り、AI応答を生成してHTMLを返す
    
    【重要】この関数が返すHTMLは、HTMXによって自動的に
    index.html の #chat-history 要素に挿入されます
    (hx-target="#chat-history" で指定されているため）
    """
    with Session(engine) as session:
        # 過去10件のメッセージを取得（会話の文脈を保つため）
        past_messages = _get_recent_messages(session, limit=10)
        
        # LLMに送るプロンプトを構築
        messages = _build_messages(past_messages, prompt)
        
        # AI応答を生成
        ai_answer = _generate_ai_response(messages)
        
        # データベースに保存
        _save_messages(session, prompt, ai_answer)
        
        # HTMLを生成して返す（HTMXがこれを #chat-history に挿入する）
        return HTMLResponse(content=_create_message_html(prompt, ai_answer))

def _get_recent_messages(session: Session, limit: int = 10) -> list[ChatMessage]:
    """データベースから最新のメッセージを取得"""
    statement = select(ChatMessage).order_by(desc(ChatMessage.id)).limit(limit)
    messages = session.exec(statement).all()
    messages.reverse()  # 古い順に並び替え
    return messages

def _build_messages(past_messages: list[ChatMessage], user_prompt: str) -> list[dict]:
    """LLMに送るメッセージリストを構築"""
    system_prompt = "あなたは親身な心理カウンセラーです。ユーザーの悩みに対して適切なアドバイスを早急かつ簡潔に教えてください。"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 過去のメッセージを追加
    for msg in past_messages:
        messages.append({"role": msg.role, "content": msg.content})
    
    # 現在のユーザー入力を追加
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def _generate_ai_response(messages: list[dict]) -> str:
    """LLMを使ってAI応答を生成"""
    response = llm.create_chat_completion(messages=messages, temperature=0.7)
    return response["choices"][0]["message"]["content"]


def _save_messages(session: Session, user_prompt: str, ai_answer: str) -> None:
    """ユーザー入力とAI応答をデータベースに保存"""
    session.add(ChatMessage(role="user", content=user_prompt))
    session.add(ChatMessage(role="assistant", content=ai_answer))
    session.commit()


def _create_message_html(user_prompt: str, ai_answer: str) -> str:
    """
    ユーザーメッセージとAI応答をHTML形式で生成

    【重要】このHTMLは、HTMXによって自動的に
    index.html の #chat-history 要素の最後に追加されます
    """
    html_content = f"""
        <div class="chat chat-end">
            <div class="chat-header text-xs opacity-50 mb-1">あなた</div>
            <div class="chat-bubble chat-bubble-info">{user_prompt}</div>
        </div>
        <div class="chat chat-start">
            <div class="chat-header text-xs opacity-50 mb-1">AI</div>
            <div class="chat-bubble chat-bubble-primary text-white">{ai_answer}</div>
        </div>
        """
    return html_content