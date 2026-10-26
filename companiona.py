"""Daras: asistente conversacional e investigador portable con fuentes abiertas."""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import sqlite3
import threading
import urllib.parse
import urllib.request
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
DB_PATH = APP_DIR / "biblioteca.sqlite3"
PREFS_PATH = APP_DIR / "config.json"
API = "https://es.wikipedia.org/w/api.php"
USER_AGENT = "Daras/0.2 (portable personal research assistant)"
BRAIN_DIR = (Path(getattr(sys, "_MEIPASS", APP_DIR)) / "cerebro"
             if getattr(sys, "frozen", False) else APP_DIR / "cerebro")


def load_cerebro():
    """Load Damian's bundled NumPy model and its safe calculator tool."""
    if str(BRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(BRAIN_DIR))
    from agent.controlled_agent import ControlledAgent
    from chat import Conversacion
    from core.checkpoint import cargar_checkpoint
    from memory.advanced_memory import AdvancedMemoryManager

    checkpoint = cargar_checkpoint(str(BRAIN_DIR / "checkpoints" / "best.npz"), con_optimizador=False)
    memory = AdvancedMemoryManager(filepath=str(APP_DIR / "daras_cerebro_memoria.json"))
    memory.guardar_largo_plazo("nombre_asistente", "Daras", categoria="persona", importancia=5)
    agent = ControlledAgent(permisos={"CALCULATE"})
    conversation = Conversacion(checkpoint["modelo"], memory, agent,
                                info_checkpoint=checkpoint["meta"], seed=None)
    conversation.gen["max_nuevos"] = 48
    memory.agregar_contexto("ia", "Soy Daras, tu compañero curioso, cercano y directo. Respondo en español.")
    return conversation


def request_json(params: dict, api: str = API) -> dict:
    url = api + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS pages (
            pageid INTEGER PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
            extract TEXT NOT NULL, saved_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT NOT NULL,
            text TEXT NOT NULL, created_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, note TEXT NOT NULL,
            created_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS research_results (
            url TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL,
            extract TEXT NOT NULL, saved_at TEXT NOT NULL)""")


def remember_turn(role: str, text: str):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("INSERT INTO conversations(role, text, created_at) VALUES (?, ?, ?)",
                   (role, text, datetime.now().astimezone().isoformat(timespec="minutes")))


def save_memory(note: str):
    with sqlite3.connect(DB_PATH) as db:
        cursor = db.execute("INSERT INTO memories(note, created_at) VALUES (?, ?)",
                            (note, datetime.now().astimezone().isoformat(timespec="minutes")))
        return cursor.lastrowid


def relevant_memories(query: str) -> list[str]:
    terms = [t for t in re.findall(r"[\wáéíóúüñ]+", query.lower()) if len(t) > 3]
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT note FROM memories ORDER BY id DESC LIMIT 100").fetchall()
    ranked = sorted(((sum(note.lower().count(t) for t in terms), note) for (note,) in rows), reverse=True)
    return [note for score, note in ranked[:3] if score > 0]


def prune_if_low_space() -> str | None:
    """Free disk only when below 512 MB; personal memories are never pruned."""
    try:
        if shutil.disk_usage(APP_DIR).free >= 512 * 1024 * 1024:
            return None
        with sqlite3.connect(DB_PATH) as db:
            chats = db.execute("DELETE FROM conversations WHERE julianday(created_at) < julianday('now','-30 days')").rowcount
            pages = db.execute("DELETE FROM pages WHERE julianday(saved_at) < julianday('now','-90 days')").rowcount
            db.execute("DELETE FROM research_results WHERE julianday(saved_at) < julianday('now','-90 days')")
            db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
        return f"Limpieza por poco espacio: se eliminaron {chats} conversaciones antiguas y {pages} artículos de caché. Tus recuerdos personales se conservaron."
    except (OSError, sqlite3.Error):
        return "No pude completar la limpieza automática; tus recuerdos personales siguen intactos."


def search_wikipedia(query: str, limit: int = 5) -> list[dict]:
    data = request_json({"action": "query", "list": "search", "srsearch": query,
                         "srlimit": limit, "format": "json", "utf8": 1})
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return []
    ids = "|".join(str(hit["pageid"]) for hit in hits)
    pages = request_json({"action": "query", "pageids": ids, "prop": "extracts",
                          "explaintext": 1, "exintro": 1, "exchars": 4500,
                          "format": "json", "utf8": 1}).get("query", {}).get("pages", {})
    result = []
    for hit in hits:
        page = pages.get(str(hit["pageid"]), {})
        extract = re.sub(r"\s+", " ", page.get("extract", "")).strip()
        if extract:
            result.append({"source": "Wikipedia", "pageid": hit["pageid"], "title": page.get("title", hit["title"]),
                           "url": "https://es.wikipedia.org/?curid=" + str(hit["pageid"]),
                           "extract": extract})
    return result


def search_wikidata(query: str) -> list[dict]:
    data = request_json({"action": "wbsearchentities", "search": query, "language": "es",
                         "uselang": "es", "limit": 4, "format": "json"},
                        "https://www.wikidata.org/w/api.php")
    results = []
    for item in data.get("search", []):
        title = item.get("label", "")
        desc = item.get("description", "")
        if title and desc:
            item_url = item.get("url", "https://www.wikidata.org/wiki/" + item.get("id", ""))
            if item_url.startswith("//"):
                item_url = "https:" + item_url
            results.append({"source": "Wikidata", "title": title, "url": item_url,
                            "extract": f"{title}: {desc}"})
    return results


def search_crossref(query: str) -> list[dict]:
    data = request_json({"query.bibliographic": query, "rows": 4,
                         "select": "DOI,title,author,published-print,abstract"},
                        "https://api.crossref.org/works")
    results = []
    for item in data.get("message", {}).get("items", []):
        title = (item.get("title") or [""])[0]
        if not title:
            continue
        abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", ""))
        authors = ", ".join(" ".join(filter(None, [a.get("given"), a.get("family")])) for a in item.get("author", [])[:3])
        year = (item.get("published-print", {}).get("date-parts") or [[None]])[0][0]
        doi = item.get("DOI", "")
        url = "https://doi.org/" + doi if doi else item.get("URL", "https://api.crossref.org/")
        extract = " · ".join(x for x in [str(year) if year else "", authors, abstract] if x)
        results.append({"source": "Crossref", "title": title, "url": url, "extract": extract or title})
    return results


def search_openalex(query: str) -> list[dict]:
    data = request_json({"search": query, "per-page": 4}, "https://api.openalex.org/works")
    results = []
    for item in data.get("results", []):
        title = item.get("title", "")
        if not title:
            continue
        doi = item.get("doi")
        url = doi or item.get("id", "https://openalex.org/")
        year = item.get("publication_year")
        abstract = ""
        inv = item.get("abstract_inverted_index") or {}
        if inv:
            words = [""] * (max(pos for positions in inv.values() for pos in positions) + 1)
            for word, positions in inv.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words)[:1800]
        results.append({"source": "OpenAlex", "title": title, "url": url,
                        "extract": (f"{year}. " if year else "") + (abstract or title)})
    return results


def search_web(query: str) -> list[dict]:
    """Use DuckDuckGo's public instant-answer endpoint, which needs no API key."""
    data = request_json({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1, "kl": "es-es"},
                        "https://api.duckduckgo.com/")
    results = []
    abstract = data.get("AbstractText", "").strip()
    if abstract:
        results.append({"source": "DuckDuckGo", "title": data.get("Heading") or query,
                        "url": data.get("AbstractURL") or "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(query),
                        "extract": abstract})
    def add_topics(items):
        for item in items or []:
            if "Topics" in item:
                add_topics(item["Topics"])
            elif item.get("Text") and item.get("FirstURL"):
                results.append({"source": "DuckDuckGo", "title": item["Text"].split(" - ", 1)[0],
                                "url": item["FirstURL"], "extract": item["Text"]})
    add_topics(data.get("RelatedTopics"))
    for item in data.get("Results", []):
        if item.get("Text") and item.get("FirstURL"):
            results.append({"source": "DuckDuckGo", "title": item["Text"].split(" - ", 1)[0],
                            "url": item["FirstURL"], "extract": item["Text"]})
    return results[:6]


def research_online(query: str) -> tuple[list[dict], list[str]]:
    """Search several open sources concurrently and retain attribution URLs."""
    sources = [search_web, search_wikipedia, search_wikidata]
    if re.search(r"\b(estudio|investigaci[oó]n|cient[ií]fic[oa]|art[ií]culo|paper|publicaci[oó]n|acad[eé]mic[oa]|evidencia|medicina|salud)\b", query, re.I):
        sources.extend([search_crossref, search_openalex])
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        jobs = {pool.submit(fn, query): fn.__name__ for fn in sources}
        for job in as_completed(jobs, timeout=35):
            name = jobs[job]
            try:
                results.extend(job.result())
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    unique = {}
    for item in results:
        unique.setdefault(item["url"], item)
    return list(unique.values())[:18], errors


def save_research_results(results: list[dict]):
    now = datetime.now().astimezone().isoformat(timespec="minutes")
    with sqlite3.connect(DB_PATH) as db:
        for item in results:
            if item.get("pageid"):
                db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?)",
                           (item["pageid"], item["title"], item["url"], item["extract"], now))
            db.execute("INSERT OR REPLACE INTO research_results VALUES (?, ?, ?, ?, ?)",
                       (item["url"], item.get("source", "Internet"), item["title"], item["extract"], now))


def save_pages(pages: list[dict]):
    with sqlite3.connect(DB_PATH) as db:
        for page in pages:
            db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?)",
                       (page["pageid"], page["title"], page["url"], page["extract"],
                        datetime.now().astimezone().isoformat(timespec="minutes")))


def find_local(query: str, limit: int = 8) -> list[dict]:
    terms = [term for term in re.findall(r"[\wáéíóúüñ]+", query.lower()) if len(term) > 2]
    if not terms:
        return []
    clauses = " OR ".join("lower(title || ' ' || extract) LIKE ?" for _ in terms)
    params = [f"%{term}%" for term in terms]
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        page_clauses = " OR ".join("lower(title || ' ' || extract) LIKE ?" for _ in terms)
        result_clauses = " OR ".join("lower(title || ' ' || extract) LIKE ?" for _ in terms)
        rows = db.execute(f"""SELECT title, url, extract, 'Wikipedia' AS source, saved_at FROM pages
                              WHERE {page_clauses}
                              UNION ALL
                              SELECT title, url, extract, source, saved_at FROM research_results
                              WHERE {result_clauses}
                              ORDER BY saved_at DESC LIMIT ?""",
                          (*params, *params, limit)).fetchall()
    return [dict(row) for row in rows]


def answer_from_pages(query: str, pages: list[dict]) -> str:
    if not pages:
        return ("Todavía no encuentro una respuesta en mi biblioteca. Si autorizas Internet, buscaré varias fuentes y compararé lo que dicen; "
                "si no, cuéntame un poco más de contexto y trataré de ayudarte con lo que ya sé.")
    terms = [t for t in re.findall(r"[\wáéíóúüñ]+", query.lower()) if len(t) > 3]
    scored = []
    for page in pages:
        for sentence in re.split(r"(?<=[.!?])\s+", page["extract"]):
            score = sum(sentence.lower().count(term) for term in terms)
            if score:
                scored.append((score, sentence.strip(), page))
    if not scored:
        scored = [(0, page["extract"][:500], page) for page in pages]
    scored.sort(key=lambda item: item[0], reverse=True)
    seen, lines = set(), []
    for _, sentence, page in scored:
        sentence = sentence[:600]
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"• {sentence} [{page['title']}]")
        if len(lines) == 4:
            break
    refs = "\n".join(f"[{p.get('source', 'Biblioteca')}] {p['title']} — {p['url']}" for p in pages[:8])
    return "Daras: Encontré estos pasajes relacionados. Son extractos de las fuentes, no una conclusión automática; puedes abrir los enlaces para ver el contexto completo.\n\n" + "\n\n".join(lines) + "\n\nFuentes consultadas:\n" + refs


def conversational_reply(query: str) -> str | None:
    text = re.sub(r"[^\wáéíóúüñ ]", "", query.lower()).strip()
    if re.fullmatch(r"(hola|holaa+|buenas|buenos dias|buenas tardes|buenas noches|hey|que tal)", text):
        return "¡Hola! Soy Daras. ¿Qué tal te está yendo hoy? Si vienes con algo en la cabeza, cuéntamelo como te salga; luego ordenamos el lío, que suele ser más útil que fingir que no existe."
    if any(phrase in text for phrase in ("como estas", "que haces", "estas ahi")):
        return "Aquí estoy, atento y sin café, una injusticia técnica que pienso seguir denunciando. ¿Qué tienes entre manos?"
    if any(word in text for word in ("gracias", "te agradezco")):
        return "De nada. Para eso estoy. ¿Quieres que sigamos con algo concreto?"
    if text in ("ayuda", "necesito ayuda", "me siento mal", "estoy triste", "estoy agobiado", "estoy agobiada"):
        return "Te escucho. No hace falta que lo expliques perfecto: dime qué pasó, qué es lo que más te pesa ahora y qué necesitas decidir. Empezamos por una cosa, no por arreglar el universo antes de cenar."
    if text in ("adios", "hasta luego", "nos vemos", "buenas noches"):
        return "Hasta luego. Deja la próxima decisión un poco más clara de como la encontraste y ya habremos avanzado."
    return None


def is_video_query(query: str) -> bool:
    return bool(re.search(r"\b(v[ií]deos?|youtube|documentales?|canales?)\b", query, re.I)
                and re.search(r"\b(recomend\w*|recom[ií]end\w*|sugier\w*|busc\w*|encuentr\w*|quiero\s+ver|ver)\b", query, re.I))


def video_topic_from_recent(query: str) -> str:
    topic = re.sub(r"\b(?:me\s+)?(?:recomiend\w*|recomend\w*|sugier\w*|busc\w*|encuentr\w*|quiero\s+ver|v[ií]deos?|youtube|documentales?|canales?|sobre|de|relacionad\w*)\b", " ", query, flags=re.I)
    topic = re.sub(r"\s+", " ", topic).strip(" ,.:;?¿¡!")
    if len(topic) > 2:
        return topic
    with sqlite3.connect(DB_PATH) as db:
        prior = db.execute("SELECT text FROM conversations WHERE role='user' ORDER BY id DESC LIMIT 8").fetchall()
    skip = {query.casefold()}
    for (text,) in prior:
        if text.casefold() in skip or is_video_query(text) or conversational_reply(text):
            continue
        if len(text.split()) >= 2:
            topic = re.sub(r"^(?:qu[eé]\s+sabes\s+de|qu[eé]\s+es|cu[eé]ntame\s+(?:sobre|de)|h[aá]blame\s+de|investiga\s+(?:sobre|acerca\s+de))\s+", "", text, flags=re.I)
            return topic or text
    return ""


def is_calculation_query(query: str) -> bool:
    """Route calculations to Cerebro's deterministic, safe calculator first."""
    try:
        if str(BRAIN_DIR) not in sys.path:
            sys.path.insert(0, str(BRAIN_DIR))
        from agent.intents import detectar_intencion
        plan = detectar_intencion(query)
        return bool(plan and plan[0].get("tool") == "calculator")
    except Exception:
        return bool(re.search(r"\d", query) and re.search(r"[+*/×÷^]|\b(m[aá]s|menos|por|entre)\b", query, re.I))


def is_out_of_corpus_scope(query: str) -> bool:
    return bool(re.search(r"\b(historia|hist[oó]ric[oa]|asesinat\w*|crimen\w*|sucesos|roblox|pel[ií]culas?|series|deportes|f[uú]tbol|pol[ií]tic\w*|noticias|actualidad)\b", query, re.I))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        init_db()
        self.title("Daras — Mentor personal")
        self.configure(bg="#07111f")
        self.geometry("900x680")
        self.minsize(720, 520)
        self._set_dark_theme()
        self.busy = False
        self.brain = None
        self.brain_error = None
        self.link_counter = 0
        self._load_prefs()
        self.internet_var = tk.BooleanVar(value=bool(self._prefs.get("internet_allowed", False)))
        self.collect_var = tk.BooleanVar(value=bool(self._prefs.get("collect_enabled", False)))
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(30000, self._auto_tick)
        self._say("Daras", "Hola. Estoy aquí para conversar, resolver cuentas, explorar curiosidades y ayudarte a encontrar cosas interesantes. Si no sé algo, prefiero reconocerlo; con tu permiso también puedo investigar y mostrarte las fuentes.")
        self.status.set("Cargando tu modelo Cerebro…")
        threading.Thread(target=self._load_brain_worker, daemon=True).start()

    def _set_dark_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#07111f", foreground="#e6edf7", fieldbackground="#0b1728")
        style.configure("TFrame", background="#07111f")
        style.configure("TLabel", background="#07111f", foreground="#d7e2f1")
        style.configure("TLabelframe", background="#07111f", foreground="#6fb5ff", bordercolor="#263b57")
        style.configure("TLabelframe.Label", background="#07111f", foreground="#70b8ff")
        style.configure("TCheckbutton", background="#07111f", foreground="#d7e2f1")
        style.map("TCheckbutton", background=[("active", "#10223a")], foreground=[("active", "#ffffff")])
        style.configure("TButton", background="#12365d", foreground="#f5f9ff", bordercolor="#246bb0", padding=(12, 7))
        style.map("TButton", background=[("active", "#1b548b"), ("disabled", "#18283a")],
                  foreground=[("disabled", "#8190a3")])
        style.configure("TEntry", background="#0b1728", foreground="#f2f6fc", insertcolor="#ffffff",
                        bordercolor="#294664", lightcolor="#294664", darkcolor="#294664", padding=7)
        style.configure("TSpinbox", background="#0b1728", foreground="#f2f6fc", arrowcolor="#70b8ff",
                        bordercolor="#294664", lightcolor="#294664", darkcolor="#294664")
        style.configure("TScrollbar", background="#13243a", troughcolor="#07111f", arrowcolor="#70b8ff")

    def _build_ui(self):
        top = ttk.Frame(self, padding=(14, 12))
        top.pack(fill="x")
        ttk.Label(top, text="DARAS", font=("TkDefaultFont", 17, "bold"), foreground="#71baff").pack(side="left")
        ttk.Label(top, text="  ·  mentor personal", font=("TkDefaultFont", 10)).pack(side="left", pady=(5, 0))
        ttk.Checkbutton(top, text="Dar acceso a Internet para investigar", variable=self.internet_var,
                        command=self._save_prefs).pack(side="right")
        self.chat = scrolledtext.ScrolledText(self, wrap="word", state="disabled", font=("TkDefaultFont", 11))
        self.chat.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.chat.configure(background="#091522", foreground="#e4edf8", insertbackground="#ffffff",
                            selectbackground="#17558c", selectforeground="#ffffff", relief="flat",
                            borderwidth=1, highlightthickness=1, highlightbackground="#1b3552",
                            padx=13, pady=12)
        self.chat.tag_configure("user", foreground="#72bdff", font=("TkDefaultFont", 11, "bold"))
        self.chat.tag_configure("assistant", foreground="#7ee0c3", font=("TkDefaultFont", 11, "bold"))
        self.chat.tag_configure("small", foreground="#9aaabe", font=("TkDefaultFont", 9))
        row = ttk.Frame(self, padding=(12, 0, 12, 8))
        row.pack(fill="x")
        self.entry = ttk.Entry(row, font=("TkDefaultFont", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.entry.bind("<Return>", lambda _event: self._ask())
        self.ask_button = ttk.Button(row, text="Preguntar / investigar", command=self._ask)
        self.ask_button.pack(side="left", padx=(8, 0))
        controls = ttk.LabelFrame(self, text="Recopilación automática autorizada", padding=10)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Checkbutton(controls, text="Recopilar estos temas mientras Daras está abierto (requiere permiso de Internet)",
                        variable=self.collect_var, command=self._save_prefs).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 7))
        ttk.Label(controls, text="Temas (uno por línea):").grid(row=1, column=0, sticky="nw")
        self.topics = tk.Text(controls, height=3, width=48, wrap="word")
        self.topics.configure(background="#0b1728", foreground="#e4edf8", insertbackground="#ffffff",
                              selectbackground="#17558c", relief="flat", highlightthickness=1,
                              highlightbackground="#294664", padx=7, pady=6)
        self.topics.grid(row=1, column=1, sticky="ew", padx=8)
        self.topics.insert("1.0", self._prefs.get("topics", "ciencia\nhistoria\ntecnología"))
        ttk.Label(controls, text="Buscar cada (minutos):").grid(row=1, column=2, sticky="e")
        self.interval = ttk.Spinbox(controls, from_=30, to=10080, increment=30, width=7)
        self.interval.grid(row=1, column=3, padx=(6, 10))
        self.interval.set(str(self._prefs.get("interval", 360)))
        controls.columnconfigure(1, weight=1)
        self.status = tk.StringVar(value="Listo. Los datos se guardan junto a esta aplicación.")
        ttk.Label(self, textvariable=self.status, relief="flat", anchor="w", padding=(10, 6),
                  background="#0b1728", foreground="#8fa8c4").pack(fill="x", side="bottom")

    def _say(self, who: str, message: str):
        remember_turn("user" if who == "Tú" else "assistant", message)
        self.chat.configure(state="normal")
        tag = "user" if who == "Tú" else "assistant"
        self.chat.insert("end", who + "\n", tag)
        position = 0
        for match in re.finditer(r"https?://[^\s<>]+", message):
            self.chat.insert("end", message[position:match.start()])
            url = match.group(0).rstrip(".,;:!?)\"]}")
            self.link_counter += 1
            link_tag = f"link_{self.link_counter}"
            self.chat.tag_configure(link_tag, foreground="#70b8ff", underline=True)
            self.chat.tag_bind(link_tag, "<Button-1>", lambda _event, target=url: webbrowser.open(target))
            self.chat.tag_bind(link_tag, "<Enter>", lambda _event: self.chat.configure(cursor="hand2"))
            self.chat.tag_bind(link_tag, "<Leave>", lambda _event: self.chat.configure(cursor=""))
            self.chat.insert("end", url, link_tag)
            position = match.start() + len(url)
        self.chat.insert("end", message[position:] + "\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _load_brain_worker(self):
        try:
            brain = load_cerebro()
            with sqlite3.connect(DB_PATH) as db:
                stored = db.execute("SELECT id, note FROM memories ORDER BY id").fetchall()
            for memory_id, note in stored:
                key = f"recuerdo_{memory_id}"
                if key not in brain.memoria.long_term_memory:
                    brain.memoria.guardar_largo_plazo(key, note, categoria="usuario", importancia=3)
            self.brain = brain
            self.after(0, lambda: self.status.set("Daras está listo. El modelo Cerebro funciona sin conexión."))
        except Exception as exc:
            self.brain_error = str(exc)
            self.after(0, lambda error=str(exc): self.status.set(f"El modelo no pudo cargar: {error}"))

    def _answer_with_brain(self, query: str):
        if self.brain is None:
            return conversational_reply(query) or "Estoy cargando el modelo local. Prueba de nuevo en un momento."
        try:
            answer, _quit = self.brain.procesar(query)
            answer = re.sub(r"\bCerebro\b", "Daras", answer, flags=re.IGNORECASE)
            return answer or "No me salió una respuesta. ¿Puedes plantearlo con otras palabras?"
        except Exception as exc:
            return f"No pude generar la respuesta con Cerebro: {exc}"

    def _ask(self):
        query = self.entry.get().strip()
        if not query or self.busy:
            return
        self.entry.delete(0, "end")
        self._say("Tú", query)
        smalltalk = conversational_reply(query)
        if smalltalk:
            normalized_short = re.sub(r"[^\wáéíóúüñ ]", "", query.casefold()).strip()
            is_greeting = bool(re.fullmatch(r"(hola|holaa+|buenas|buenos dias|buenas tardes|buenas noches|hey|que tal|como estas|que haces|estas ahi|gracias|muchas gracias)", normalized_short))
            answer = self._answer_with_brain(query) if is_greeting and self.brain else smalltalk
            self._say("Daras", answer)
            return
        normalized = query.casefold()
        access_request = ("internet" in normalized and ":" not in query
                          and (any(word in normalized for word in ("acceso", "autoriz", "permiso", "conecta"))
                               or normalized.strip(" .,!?¿¡") in {"daras investiga por internet", "investiga por internet"})
                          and len(query.split()) < 12)
        if access_request:
            self.internet_var.set(True)
            self._save_prefs()
            self._say("Daras", "Entendido. Queda autorizado el acceso a Internet para investigar tus próximas preguntas. Buscaré en varias fuentes abiertas y te mostraré los enlaces; si alguna no responde, te lo diré.")
            return
        if query.lower().startswith("recuerda:"):
            note = query.split(":", 1)[1].strip()
            if note:
                memory_id = save_memory(note)
                if self.brain:
                    self.brain.memoria.guardar_largo_plazo(f"recuerdo_{memory_id}", note,
                                                            categoria="usuario", importancia=3)
                self._say("Daras", "Anotado. Lo conservaré como un recuerdo personal. Puedes escribir «olvida: palabra» para borrar los recuerdos que coincidan.")
            return
        if query.lower().startswith("olvida:"):
            term = query.split(":", 1)[1].strip()
            with sqlite3.connect(DB_PATH) as db:
                matching = db.execute("SELECT id FROM memories WHERE lower(note) LIKE ?", (f"%{term.lower()}%",)).fetchall() if term else []
                removed = db.execute("DELETE FROM memories WHERE lower(note) LIKE ?", (f"%{term.lower()}%",)).rowcount if term else 0
            if self.brain:
                for (memory_id,) in matching:
                    self.brain.memoria.eliminar_largo_plazo(f"recuerdo_{memory_id}")
            self._say("Daras", f"He borrado {removed} recuerdo(s) que coinciden." if removed else "No encontré recuerdos con ese texto.")
            return
        explicit_search = any(phrase in normalized for phrase in ("investiga en internet:", "busca en internet:", "consulta en internet:"))
        if explicit_search:
            query = re.split(r"(?:investiga|busca|consulta) en internet:", query, flags=re.IGNORECASE, maxsplit=1)[-1].strip()
        local = find_local(query)
        if is_calculation_query(query):
            answer = self._answer_with_brain(query)
            self._say("Daras", answer)
        elif is_video_query(query):
            topic = video_topic_from_recent(query)
            if not topic:
                self._say("Daras", "Claro. ¿Sobre qué tema quieres los vídeos? Si prefieres que busque enlaces actuales, activa el acceso a Internet o escribe «busca en internet: …».")
            elif self.internet_var.get() or explicit_search:
                self._research(topic, local, video_only=True)
            else:
                self._say("Daras", f"Puedo proponerte vídeos sobre «{topic}», pero para darte enlaces actuales necesito que actives el acceso a Internet.")
        elif self.internet_var.get() or explicit_search:
            self._research(query, local)
        elif local:
            self._say("Daras", answer_from_pages(query, local))
        elif is_out_of_corpus_scope(query):
            self._say("Daras", "Ese tema queda fuera de lo que mi pequeño modelo Cerebro aprendió. Prefiero no inventarte una respuesta. Activa Internet o escribe «investiga en internet: " + query + "» y buscaré fuentes.")
        else:
            answer = self._answer_with_brain(query)
            memories = relevant_memories(query)
            if memories:
                answer += "\n\nLo que recuerdo de ti: " + " · ".join(memories)
            cleanup = prune_if_low_space()
            if cleanup:
                answer += "\n\n" + cleanup
            self._say("Daras", answer)

    def _research(self, query: str, local: list[dict] | None = None, video_only: bool = False):
        if self.busy:
            return
        self.busy = True
        self.ask_button.configure(state="disabled")
        self.status.set("Buscando vídeos relacionados…" if video_only else "Buscando y comparando varias fuentes abiertas…")
        def worker():
            try:
                if video_only:
                    pages = search_web(f"site:youtube.com/watch {query} español vídeo")
                    pages = [p for p in pages if "youtube.com" in p["url"] or "youtu.be" in p["url"]]
                    if pages:
                        for page in pages:
                            page["source"] = "YouTube"
                        errors = []
                    else:
                        search_url = "https://www.youtube.com/results?" + urllib.parse.urlencode({"search_query": query})
                        pages = [{"source": "YouTube", "title": f"Abrir resultados de YouTube para «{query}»",
                                  "url": search_url, "extract": "No pude recuperar títulos individuales, pero aquí tienes la búsqueda directa.",
                                  "search_link": True}]
                        errors = ["No pude recuperar vídeos individuales."]
                else:
                    pages, errors = research_online(query)
                save_research_results(pages)
                self.after(0, lambda: self._research_done(query, pages, errors, video_only))
            except Exception as exc:
                self.after(0, lambda error=str(exc): self._research_done(query, local or [], [error], video_only))
        threading.Thread(target=worker, daemon=True).start()

    def _research_done(self, query: str, pages: list[dict], errors: list[str], video_only: bool = False):
        self.busy = False
        self.ask_button.configure(state="normal")
        if video_only:
            if pages and pages[0].get("search_link"):
                answer = f"No pude traer títulos individuales desde el buscador, así que te dejo la búsqueda directa en YouTube para «{query}»:\n\n{pages[0]['url']}"
                self.status.set("Enlace directo a resultados de YouTube preparado.")
            elif pages:
                answer = f"Encontré estos vídeos de YouTube relacionados con «{query}»:\n\n" + "\n\n".join(
                    f"• {p['title']}\n  {p['url']}" + (f"\n  {p['extract']}" if p.get("extract") != p['title'] else "")
                    for p in pages[:5])
                self.status.set(f"Encontré {len(pages)} vídeos relacionados.")
            else:
                answer = f"No encontré vídeos de YouTube para «{query}». Prueba con otra frase o un tema más concreto."
                self.status.set("No se encontraron vídeos.")
            self._say("Daras", answer)
            return
        if errors:
            self.status.set(f"Investigación terminada; {len(errors)} fuente(s) no respondieron.")
            answer = answer_from_pages(query, pages) if pages else (
                "No encontré resultados útiles en las fuentes disponibles; prefiero no inventarte una respuesta. "
                "Prueba con otros términos.\n\n" + "; ".join(errors[:3]))
            if pages:
                answer += "\n\nAlgunas fuentes no respondieron, así que la comparación puede estar incompleta."
        else:
            self.status.set(f"Listo. Consulté {len(pages)} resultados y los guardé en la biblioteca local.")
            answer = answer_from_pages(query, pages)
        memories = relevant_memories(query)
        if memories:
            answer += "\n\nLo que recuerdo de ti: " + " · ".join(memories)
        cleanup = prune_if_low_space()
        if cleanup:
            answer += "\n\n" + cleanup
        self._say("Daras", answer)

    def _auto_tick(self):
        self._save_prefs()
        cleanup = prune_if_low_space()
        if cleanup:
            self.status.set(cleanup)
        if self.collect_var.get() and self.internet_var.get() and not self.busy:
            prefs = self._prefs
            due = datetime.now().timestamp() - prefs.get("last_collection", 0) >= max(30, int(self.interval.get() or 360)) * 60
            topics = [line.strip() for line in self.topics.get("1.0", "end").splitlines() if line.strip()]
            if due and topics:
                topic = topics[prefs.get("topic_cursor", 0) % len(topics)]
                prefs["topic_cursor"] = prefs.get("topic_cursor", 0) + 1
                prefs["last_collection"] = datetime.now().timestamp()
                self._save_prefs()
                self.status.set(f"Recopilando en varias fuentes el tema autorizado: {topic}")
                threading.Thread(target=self._auto_worker, args=(topic,), daemon=True).start()
        self.after(30000, self._auto_tick)

    def _auto_worker(self, topic: str):
        try:
            pages, _errors = research_online(topic)
            save_research_results(pages)
            self.after(0, lambda: self.status.set(f"Recopilación terminada: {len(pages)} artículos sobre «{topic}»."))
        except Exception as exc:
            self.after(0, lambda error=str(exc): self.status.set(f"Recopilación pausada por error: {error}"))

    def _load_prefs(self):
        try:
            self._prefs = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._prefs = {}

    def _save_prefs(self):
        if not hasattr(self, "topics"):
            return
        try:
            self._prefs["topics"] = self.topics.get("1.0", "end").strip()
            self._prefs["interval"] = int(self.interval.get() or 360)
            self._prefs["internet_allowed"] = bool(self.internet_var.get())
            self._prefs["collect_enabled"] = bool(self.collect_var.get())
            PREFS_PATH.write_text(json.dumps(self._prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError, tk.TclError):
            pass

    def _close(self):
        self._save_prefs()
        self.destroy()


if __name__ == "__main__":
    if "--verificar-modelo" in sys.argv:
        try:
            brain = load_cerebro()
            answer, _ = brain.procesar("2 mas 2")
            print(answer)
            sys.exit(0 if "4" in answer else 1)
        except Exception as exc:
            print(f"No se pudo cargar Cerebro: {exc}")
            sys.exit(1)
    App().mainloop()
