

# Client-Side vs Server-Side Storage Strategy for Tabasco-FastAPI

## 🧩 Problem Recap

Tabasco-FastAPI is a single-session, transformer-based NLP web app. Users upload a PDF, which is processed in multiple interactive steps. The primary challenge is:

- **User refreshes or navigates away → session is lost**
- Server should avoid unnecessary resource usage and persistent storage
- Goal: **Minimize server load and maximize client-side storage**

---

## 🌐 Browser Storage Options

| Storage Type        | Persistent? | Max Size       | Accessible From | Use Case                                  |
|---------------------|-------------|----------------|------------------|--------------------------------------------|
| **Local Storage**   | ✅ Yes       | ~5MB           | JS only          | Small key-value state                      |
| **Session Storage** | ❌ No        | ~5MB           | JS only          | Temporary tab/session-only state           |
| **IndexedDB**       | ✅ Yes       | ~2GB+          | JS only          | Large structured/blob data (PDF, text)     |
| **Cache Storage**   | ✅ Yes       | Varies         | Service Workers  | Offline-first PWA content                  |
| **Cookies**         | ✅ Yes       | ~4KB per cookie| Backend/JS       | Auth/session ID (not for large payloads)   |

---

## 🔍 What Are Ephemeral Sessions?

- Ephemeral sessions **don’t persist** beyond the session (no long-term cookies, caches).
- In your use-case, this means:
  - **Short-lived processing** (5–30 mins)
  - **No auth or persistent identity**
  - Optionally store in memory/disk temporarily and auto-clean

---

## ✅ Options You Have

### Option 1: **Store Everything on Client (Pure Client)**

Use `pdf.js` or `pdf-lib` to extract text **in browser**.

- Pros: Zero backend processing, no server storage
- Cons: Can't use BERT/NLP models easily in-browser
- Only feasible if you avoid server-side inference

✅ **Use:** `IndexedDB` to store parsed PDF or extracted text

---

### Option 2: **Minimal Session Server (Hybrid - Recommended)**

Server:
- On PDF upload, extract text (use `pdfplumber`, etc.)
- Store to SQLite (`session_id`, `text`)
- Return `session_id` to client

Client:
- Store `session_id` in `localStorage`
- Resume session via `/api/session/<session_id>`

SQLite Example Table:

```sql
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  extracted_text TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Auto-clean with cron job:
```sql
DELETE FROM sessions WHERE created_at < DATETIME('now', '-60 minutes');
```

---

### Option 3: **Client-Centric with Server Recompute**

- Client stores everything in IndexedDB
- If session lost, re-upload and recompute
- Can also cache summaries client-side

✅ Works if NLP processing is cheap or cacheable

---

### Option 4: **Stateless One-Off (Not Recommended)**

- User uploads → server processes and responds → nothing is stored
- If page refreshes, session is lost

This is the current default behavior — it’s simple but painful for users.

---

## 🧠 Final Recommendation (Balanced)

- 🧩 Extract text on server
- 🧱 Store in SQLite (`session_id`, `text`)
- 🧹 Clean sessions >60 minutes old via background job
- 🧠 Return session_id to frontend → store in `localStorage`
- 🔁 All further interactions use `session_id`

This gives a **minimal server**, **persistent UX**, and **clean teardown**.

---

## Summary Table

| Goal                            | Recommended Approach                        |
|---------------------------------|----------------------------------------------|
| Reduce server disk use          | Don’t store PDFs, store only extracted text |
| Avoid full DBMS overhead        | Use SQLite with TTL or cron cleanup         |
| Resume on browser refresh       | Store `session_id` in LocalStorage          |
| High-speed access (advanced)    | Use Redis (optional)                        |
| Pure client-side (if possible)  | Use IndexedDB + WebWorker NLP (future idea) |

---

