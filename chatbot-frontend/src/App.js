import React, { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import "./App.css";
import ReactMarkdown from "react-markdown";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sessionId] = useState(uuidv4());
  const [loading, setLoading] = useState(false);

  // 🔹 Reset backend session on page load
  useEffect(() => {
    const resetSession = async () => {
      try {
        await fetch("http://127.0.0.1:8000/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
      } catch (err) {
        console.error("Error resetting on refresh:", err);
      }
    };
    resetSession();
  }, [sessionId]);

  // 🔹 refs for suggestion rows
  const scrollRefs = [useRef(null), useRef(null), useRef(null)];

  /* === Mouse Wheel Scroll === */
  const handleWheel = (e) => {
    e.preventDefault();
    const scrollAmount = e.deltaY !== 0 ? e.deltaY : e.deltaX;

    scrollRefs.forEach((ref) => {
      if (ref.current) {
        ref.current.scrollLeft += scrollAmount * 2;
      }
    });
  };

  /* === Touch Events === */
  let touchStartX = 0;
  const handleTouchStart = (e) => {
    touchStartX = e.touches[0].clientX;
  };
  const handleTouchMove = (e) => {
    if (!touchStartX) return;
    const touchX = e.touches[0].clientX;
    const deltaX = touchStartX - touchX;

    scrollRefs.forEach((ref) => {
      if (ref.current) {
        ref.current.scrollLeft += deltaX * 1.5;
      }
    });

    touchStartX = touchX;
  };

  /* === Typing Effect for Bot Reply === */
  const typeBotReply = (fullText) => {
    let index = 0;

    const interval = setInterval(() => {
      if (index < fullText.length) {
        setMessages((prev) => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.sender === "bot") {
            const updated = [...prev];
            updated[updated.length - 1] = {
              sender: "bot",
              text: lastMsg.text + fullText[index],
            };
            return updated;
          } else {
            return [...prev, { sender: "bot", text: fullText[index] }];
          }
        });
        index++;
      } else {
        clearInterval(interval);
      }
    }, 20); // typing speed per character
  };

  /* === Send Message === */
  const sendMessage = async (text) => {
    if (!text.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      const data = await response.json();

      // Add bot reply placeholder
      setMessages((prev) => [...prev, { sender: "bot", text: "" }]);
      typeBotReply(data.reply);

      // Show image if available
      if (data.image_url) {
        setMessages((prev) => [
          ...prev,
          { sender: "bot-image", imageUrl: data.image_url },
        ]);
      }
    } catch (err) {
      console.error("Error fetching response:", err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Error: Could not reach server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* === Prompt Suggestions === */
  const prompts = [
    "I want a shirt to impress my crush",
    "What snacks for late night gossips",
    "Memorable coffee mug for my friend",
    "What gift makes someone fall fast",
    "I need breakup recovery snacks",
    "I want flowers to give my ex",
    "What outfit to make crush jealous",
    "I need Maggi at 3am urgently",
    "What snacks for movie night",
    "I want a hangover survival kit",
    "What gift annoys my roommate",
    "I need instant glow-up products",
    "I want party outfit to flex",
    "I need speaker for late-night dance",
    "I want midnight ice cream delivery",
    "What chips for gossip session vibes",
    "I want secret date-night essentials",
    "What perfume makes ex jealous",
    "What hoodie hides last night hangover",
    "I want snacks for boring lecture",
    "I need instant mood booster",
    "Fun activities to do on a Saturday night",
  ];

  // split into 3 rows
  const rowCount = 3;
  const promptsByRow = Array.from({ length: rowCount }, (_, i) =>
    prompts.filter((_, idx) => idx % rowCount === i)
  );

  return (
    <div className="app-container">
      <div className="chat-header">
        <span className="logo clickable">✨ Acozmos AI</span>
        <button
          className="new-chat-btn"
          onClick={async () => {
            setMessages([]);
            try {
              await fetch("http://127.0.0.1:8000/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId }),
              });
            } catch (err) {
              console.error("Error resetting chat:", err);
            }
          }}
        >
          + New Chat
        </button>
      </div>

      {messages.length === 0 ? (
        <div
          className="home-screen"
          onWheel={handleWheel}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
        >
          <div className="tagline">Where shopping begins</div>

          {/* 🔹 3 rows of prompts */}
          <div className="suggestions-multirow">
            {promptsByRow.map((row, rowIdx) => (
              <div
                key={rowIdx}
                className="suggestions-scroll"
                ref={scrollRefs[rowIdx]}
              >
                {row.map((p, idx) => (
                  <div
                    key={idx}
                    className="prompt-pill"
                    onClick={() => sendMessage(p)}
                  >
                    {p}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-window">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`message ${
                msg.sender === "user"
                  ? "user-message"
                  : msg.sender === "bot"
                  ? "bot-message"
                  : "bot-image"
              }`}
            >
              {msg.sender === "bot-image" ? (
                <img
                  src={msg.imageUrl}
                  alt="Product"
                  className="chat-image"
                />
              ) : (
                <div className="markdown">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>

              )}
            </div>
          ))}

          {loading && (
            <div className="message bot-message typing">
              <span></span>
              <span></span>
              <span></span>
            </div>
          )}
        </div>
      )}

      <div className="chat-input-container">
        <input
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Need anything..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage(input)}
        />
        <button className="send-btn" onClick={() => sendMessage(input)}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
