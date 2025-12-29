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

# 2. チャット送信時の処理 (HTMLを返すように変更！)
@app.post("/chat", response_class=HTMLResponse)
def chat_endpoint(request: Request, prompt: str = Form(...)): # Form(...)でデータを受け取る
    with Session(engine) as session:
        # --- 記憶の取得とAI生成（ロジックは前回と同じ） ---
        statement = select(ChatMessage).order_by(desc(ChatMessage.id)).limit(10)
        past_messages = session.exec(statement).all()
        past_messages.reverse()

        messages = [{"role": "system", "content": "あなたは親身な心理カウンセラーです。ユーザーの悩みに対して適切なアドバイスを簡潔に教えてください。"}]
        for msg in past_messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": prompt})

        response = llm.create_chat_completion(messages=messages, temperature=0.7)
        ai_answer = response["choices"][0]["message"]["content"]

        # データベースに保存
        session.add(ChatMessage(role="user", content=prompt))
        session.add(ChatMessage(role="assistant", content=ai_answer))
        session.commit()

        # --- 【重要】ここが変わりました！ ---
        # JSONではなく、追加表示したい「HTMLの部品」を直接返します
        # --- Python側の返却HTMLをDaisyUI仕様に変更 ---
        html_content = f"""
        <div class="chat chat-end">
            <div class="chat-header text-xs opacity-50 mb-1">あなた</div>
            <div class="chat-bubble chat-bubble-info">{prompt}</div>
        </div>
        <div class="chat chat-start">
            <div class="chat-header text-xs opacity-50 mb-1">AI</div>
            <div class="chat-bubble chat-bubble-primary text-white">{ai_answer}</div>
        </div>
        """
        return HTMLResponse(content=html_content)