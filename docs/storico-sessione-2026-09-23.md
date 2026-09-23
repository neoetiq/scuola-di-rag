# Storico della sessione — KB aziendale da manuali PDF (23 settembre 2026)

Progetto: `scuola-di-rag`. Obiettivo: knowledge base aziendale (RAG) da manuali di installazione, uso e
manutenzione di stufe e caminetti, PDF multilingue, tutto in esecuzione locale (Apple M5 Pro, 48 GB RAM).
Questo documento ricostruisce in ordine cronologico le richieste, le decisioni prese e i risultati misurati.

---

## 1. Scelta dei modelli di embedding

**Richiesta.** Identificare i modelli di embedding più adatti a PDF multilingue di manuali tecnici, solo
modelli eseguibili in locale.

**Analisi.** Requisiti chiave: retrieval cross-lingua, robustezza su codici prodotto e codici errore,
contesto lungo per le tabelle, licenza commerciale.

**Shortlist proposta.**

| modello | parametri | contesto | licenza |
|---|---|---|---|
| Qwen3-Embedding-4B (e 0.6B / 8B) | 4B | 32k | Apache 2.0 |
| BGE-M3 | 568M | 8k | MIT |
| multilingual-e5-large-instruct | 560M | 512 | MIT |

Seconda fila: Granite Embedding Multilingual R2, Snowflake Arctic Embed L v2, nomic-embed-text-v2-moe,
EmbeddingGemma. Esclusi Jina v3/v4 per licenza CC-BY-NC. Indicati anche i reranker locali
(bge-reranker-v2-m3, Qwen3-Reranker) e, per le pagine grafiche, i modelli visivi (Qwen3-VL-Embedding, ColQwen).

---

## 2. Analisi dei PDF e scelta dell'estrattore

**Richiesta.** Usare i due manuali `H072510EN0 (EN)` e `H072521IT0 (IT)` per i test; software in Python con
LangChain; valutare estrattori PDF dedicati (MinerU, Watson/Docling, altri) se i PDF fossero difficili.
Poco dopo: **l'estrazione deve produrre file `.md`**.

**Cosa è emerso dai PDF.** Testo nativo InDesign, nessuna scansione, quindi niente OCR. Difficoltà di layout:
due colonne, sillabazione a fine riga, header/footer ripetuti, icone di avvertenza rese come lettere di un
font simbolico (`d` = pericolo, `a` = attenzione, `i` = info), tabella dati tecnici con celle unite, disegni quotati,
pagine di menu con schemi.

**Estrattori provati.**

| estrattore | tempo/manuale | esito |
|---|---|---|
| pymupdf4llm | ~5 s | buona gerarchia dei titoli; ordine di lettura sbagliato sulle pagine con schemi; celle unite spezzate |
| Docling (IBM) | ~70-90 s | ordine di lettura corretto, tabelle con celle unite intatte, figure sostituite da segnaposto. **Scelto come default** |
| LangChain PyPDFLoader (pypdf) | ~2 s | testo piatto, senza struttura: baseline |
| MinerU 4 | ~65 s | vedi §5 |

**Pipeline realizzata** (`src/kb/`, progetto `uv`, Python 3.12+):

- `kb-extract`: PDF → Markdown in `data/md/<backend>/`, front matter YAML (source, title, doc_code, language),
  marcatori `<!-- page: N -->`. Output grezzo del backend cachato in `data/md/raw/`.
- `clean.py`: rimozione header/footer ripetuti (per frequenza e per righe maiuscole ricorrenti), icone →
  `[!DANGER]` / `[!WARNING]` / `[!INFO]`, sillabazione, normalizzazione dei livelli dei titoli dalla numerazione,
  compattazione delle tabelle (padding, celle duplicate da celle unite, tabelle HTML → Markdown).
- `chunk.py`: split per titoli Markdown + lunghezza (1200 caratteri, overlap 150), breadcrumb di sezione,
  metadata con pagina iniziale/finale.
- `embeddings.py`: factory `hf:` (sentence-transformers su MPS) / `ollama:`.
- `index.py`: un indice Chroma per estrattore × modello. `evaluate.py`: hit@k e MRR.
- `eval/questions.jsonl`: 44 domande (poi 52) con manuale e pagine attese, incluse cross-lingua e codici allarme.
- `.gitignore` esclude `pdf/` e `data/` (documenti aziendali e dati generati).

---

## 3. Benchmark estrattore × modello (2 manuali, 44 domande)

| estrattore | modello | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| docling | Qwen3-Embedding-4B | 0.93 | 0.98 | **0.953** |
| docling | BGE-M3 | 0.91 | 0.95 | **0.940** |
| docling | Qwen3-Embedding-0.6B | 0.84 | 0.89 | 0.880 |
| docling | multilingual-e5-large-instruct | 0.77 | 0.86 | 0.830 |
| pymupdf4llm | Qwen3-Embedding-4B | 0.86 | 0.93 | 0.903 |
| pymupdf4llm | BGE-M3 | 0.84 | 0.91 | 0.889 |
| pymupdf4llm | Qwen3-Embedding-0.6B | 0.77 | 0.89 | 0.848 |
| pymupdf4llm | multilingual-e5-large-instruct | 0.68 | 0.77 | 0.746 |

Osservazioni: l'estrattore pesa quanto il modello (Docling +4-8 punti di MRR su tutti); Qwen3-4B e BGE-M3
quasi equivalenti, BGE-M3 7× più piccolo e 4× più veloce; e5-large soffre su cross-lingua e codici.

---

## 4. Domanda: Qwen3-Embedding-4B supporta i vettori sparse?

No: è un modello solo dense. BGE-M3 produce dense, sparse e multi-vector dallo stesso passaggio. Per la ricerca
ibrida con Qwen3 serve un componente lessicale separato (BM25).

---

## 5. MinerU e LangChain PyPDFLoader nella comparazione

**Richiesta.** Aggiungere MinerU (con Qwen3-4B e BGE-M3) e poi anche l'estrattore standard di LangChain.

**MinerU 4.0.6.** È diventato un "document center" con server locale e servizio remoto su mineru.net. Di
default il parser locale è disabilitato e il CLI propone il remoto: escluso per documenti aziendali. Setup fatto:

```bash
uv run mineru-kit models download --tier standard    # ~2 GB (layout ONNX + VLM MinerU2.5 1.2B GGUF)
uv run mineru config set parse_server.local.mode managed
uv run mineru telemetry disable
uv run mineru server start
```

Il backend `mineru` chiama `mineru parse --pages all --tier standard` sul server locale e divide il Markdown sui
marcatori `<!-- page N of M -->`. MinerU ha creato una cartella `blobs/` nella root del repo (aggiunta a `.gitignore`).

**Domanda intermedia: cosa sono le "tabelle ONNX".** ONNX è il formato standard per reti neurali già addestrate,
eseguite da ONNX Runtime su CPU senza PyTorch. MinerU usa tre modelli ONNX per le tabelle: PP-LCNet (classifica
tabelle con/senza bordi), SLANet-plus (struttura righe/colonne/celle unite → HTML), UNet (segmentazione linee).
Il lavoro pesante (testo e layout) lo fa il VLM su Metal via llama.cpp.

**Risultati (2 manuali, 44 domande, catene dense).**

| estrattore | modello | hit@1 | hit@3 | MRR | s/query |
|---|---|---|---|---|---|
| docling | Qwen3-4B | 0.93 | 0.98 | 0.953 | 0.066 |
| docling | BGE-M3 | 0.91 | 0.95 | 0.940 | 0.017 |
| mineru | Qwen3-4B | 0.91 | 0.93 | 0.930 | 0.068 |
| pymupdf4llm | Qwen3-4B | 0.86 | 0.93 | 0.903 | 0.053 |
| mineru | BGE-M3 | 0.84 | 0.91 | 0.891 | 0.020 |
| langchain (PyPDFLoader) | Qwen3-4B | 0.82 | 0.95 | 0.890 | 0.053 |
| pymupdf4llm | BGE-M3 | 0.84 | 0.91 | 0.889 | 0.016 |
| langchain (PyPDFLoader) | BGE-M3 | 0.70 | 0.86 | 0.787 | 0.016 |

Conclusioni: Docling resta il migliore; MinerU vicino con Qwen3-4B; PyPDFLoader perde 15 punti con BGE-M3 e 6
con Qwen3-4B; Qwen3-4B è più tollerante al testo senza struttura.

---

## 6. Strumento interattivo `kb-query`

**Richiesta.** CLI interattiva: chiede una domanda, mostra i primi 3 chunk, ripete; `quit` per uscire.

Realizzato in `src/kb/interactive.py`: carica l'indice una volta (`--extractor`, `--model`, `-k`), stampa
punteggio, manuale, pagina, breadcrumb e testo del chunk. Accetta `quit`, `exit`, `q`, `esci`, Ctrl-D.

---

## 7. Vector DB: Chroma e passaggio a Qdrant

**Domanda.** Quale vector DB è in uso? Chroma, embedded, un indice per estrattore × modello, metrica coseno,
metadata per chunk. Suggerito Qdrant per la KB reale: vettori sparse nativi, filtri sui metadata scalabili.

**Richiesta.** Dato che i documenti contengono molti codici (anche di errore), usare anche vettori sparse e
full-text/BM25: opzioni e collegamento al vector DB.

**Opzioni illustrate.** BM25 in memoria; full text del DB (FTS5/tsvector); sparse appresi (BGE-M3 sparse,
SPLADE); BM25 e SPLADE dentro Qdrant via FastEmbed. Raccomandazione: BGE-M3 dense + sparse in Qdrant, fusione
RRF, reranker locale; per Qwen3 dense, BM25 come ramo lessicale.

**Modelli sparse che supportano l'italiano.** BGE-M3 sparse (multilingue, 100+ lingue) è l'unico maturo.
OpenSearch neural-sparse multilingual v1 non dichiara l'italiano tra le 15 lingue. SPLADE italiano
(nickprock) è un modello di comunità poco usato. SPLADE++, BM42, MiniCOIL: solo inglese. BM25 senza stemming è
indipendente dalla lingua.

---

## 8. Benchmark ibrido su Qdrant

**Richiesta.** Confrontare la catena "full BGE" (dense + sparse + RRF + reranking) e "Qwen3-4B + BM25 + SPLADE +
reranker" dentro Qdrant; poi anche BGE-M3 sparse nella catena B.

**Implementazione** (`src/kb/sparse.py`, `src/kb/hybrid.py`, comando `kb-hybrid`):

- Qdrant embedded (`data/qdrant/<estrattore>/`, nessun server); verificato che la modalità locale supporta più
  vettori sparse e la fusione RRF.
- Encoder BGE-M3 dense + sparse in un solo passaggio (transformers + testa `sparse_linear.pt`); il dense
  coincide con sentence-transformers (differenza massima 0.0002).
- BM25 (`Qdrant/bm25`, senza stemming) e SPLADE (`prithivida/Splade_PP_en_v1`, solo inglese) via fastembed.
- Reranker: `BAAI/bge-reranker-v2-m3` (CrossEncoder) e, aggiunto in seguito, `Qwen/Qwen3-Reranker-0.6B`.
- Due collection (dense BGE-M3 e dense Qwen3-4B), ciascuna con `bge_sparse`, `bm25`, `splade` per chunk.
- 11 catene: A0 bge dense; A1 +sparse RRF; A2 +reranker; B0 qwen dense; B1 +bm25+splade RRF; B2 +reranker;
  B1m qwen+bm25+bge_sparse; B2m +reranker; S1/S2/S3 solo sparse.
- Set di domande portato a 52 (8 domande in più su codici e identificativi).

**Risultati su 2 manuali (316 chunk).** Dense da solo già al limite (A0 0.939, B0 0.941). I rami sparse portano
i codici a MRR 1.000 ma penalizzano cross-lingua e tabelle; il reranker non aiuta.

---

## 9. Ingestione di tutti i PDF e benchmark sul corpus completo

**Richiesta.** Ingerire tutti i documenti della cartella `pdf` prima dei test; tenere aggiornato l'avanzamento.

**Ingestione.** Al momento dell'ingestione la cartella conteneva 13 PDF (all'inizio della sessione erano 16: i
manuali FR, NL e PL non erano più presenti). Estrazione Docling di tutti i 13, ~70 s l'uno: 1935 chunk.
Ricostruite le collection Qdrant: BGE-M3 dense+sparse 18 s, Qwen3-4B dense 302 s, BM25+SPLADE 49 s.

**Problema di ambiguità.** I 9 manuali italiani condividono capitoli identici parola per parola (presa d'aria,
pulizia del vetro, manutenzione, distanze di sicurezza). Senza indicare il prodotto molte domande sono ambigue e
il punteggio stretto le conta come errori. Aggiunti al benchmark:

- campo `product` in ogni domanda;
- modalità `product` (nome del prodotto appeso alla domanda) e `filter` (filtro Qdrant sul `doc_code`);
- metrica `fam` (stesso manuale in un'altra lingua conta come corretto).

**Risultati, modalità `plain` (domanda nuda).**

| catena | hit@1 | hit@3 | MRR | codici | cross | tabelle |
|---|---|---|---|---|---|---|
| A0 BGE-M3 dense | 0.52 | 0.67 | 0.620 | 0.843 | 0.710 | 0.330 |
| A1 BGE-M3 dense+sparse RRF | 0.54 | 0.67 | 0.628 | 0.903 | 0.658 | 0.391 |
| A2 A1 + bge-reranker | 0.48 | 0.63 | 0.596 | 0.875 | 0.579 | 0.272 |
| B0 Qwen3-4B dense | 0.62 | 0.73 | 0.706 | 0.875 | 0.909 | 0.439 |
| B1 Qwen3-4B+BM25+SPLADE RRF | 0.56 | 0.63 | 0.640 | 0.938 | 0.652 | 0.428 |
| B2 B1 + bge-reranker | 0.48 | 0.71 | 0.623 | 0.861 | 0.609 | 0.335 |
| B1m Qwen3-4B+BM25+BGE-M3 sparse RRF | 0.60 | 0.67 | 0.674 | 0.958 | 0.772 | 0.452 |
| B2m B1m + bge-reranker | 0.48 | 0.71 | 0.621 | 0.861 | 0.609 | 0.326 |
| solo BGE-M3 sparse / BM25 / SPLADE | 0.38 / 0.37 / 0.40 | | 0.509 / 0.471 / 0.497 | | | |

**Verifica del reranker.** bge-reranker-v2-m3 peggiorava anche con il filtro sul manuale. Verificato che non è un
problema numerico (fp16 = fp32 = CPU). Casi concreti: preferisce la prosa sullo "scarico fumi" alla riga della
tabella con "Diametro scarico fumi 80 mm", e il paragrafo che rimanda alla procedura invece della procedura.
Qwen3-Reranker-0.6B risolve entrambi i casi (0.99 alla riga di tabella) ed è stato aggiunto come opzione `--reranker qwen`.

**Risultati, modalità `product` e `filter`.**

| catena | MRR product | hit@3 product | MRR filter | hit@1 filter | codici filter | s/query |
|---|---|---|---|---|---|---|
| A0 BGE-M3 dense | 0.864 | 0.94 | **0.975** | **0.96** | 1.000 | 0.08 |
| A1 BGE-M3 dense+sparse RRF | 0.831 | 0.94 | 0.969 | 0.96 | 1.000 | 0.10 |
| A2 A1 + bge-reranker | 0.839 | 0.88 | 0.903 | 0.83 | 0.861 | 0.42 |
| A2 A1 + Qwen3-Reranker | 0.870 | 0.98 | 0.913 | 0.85 | 0.958 | 1.08 |
| B0 Qwen3-4B dense | 0.857 | 0.94 | 0.948 | 0.92 | 0.933 | 0.07 |
| B1 Qwen3-4B+BM25+SPLADE RRF | 0.756 | 0.88 | 0.944 | 0.90 | 1.000 | 0.14 |
| B2 B1 + bge-reranker | 0.854 | 0.94 | 0.885 | 0.81 | 0.861 | 0.38 |
| B2 B1 + Qwen3-Reranker | **0.891** | **1.00** | 0.908 | 0.85 | 0.958 | 1.27 |
| B1m Qwen3-4B+BM25+BGE-M3 sparse RRF | 0.774 | 0.92 | 0.897 | 0.85 | 1.000 | 0.13 |
| B2m B1m + bge-reranker | 0.853 | 0.94 | 0.884 | 0.81 | 0.861 | 0.37 |
| B2m B1m + Qwen3-Reranker | 0.878 | 1.00 | 0.908 | 0.85 | 0.958 | 1.32 |

**Conclusioni.**

1. Il filtro sul manuale è la leva più forte: da 0.86 a 0.975. L'applicazione deve far scegliere il prodotto o
   riconoscerlo dalla domanda.
2. Con il filtro, BGE-M3 dense da solo è imbattibile; sparse e reranker non aggiungono nulla.
3. Senza filtro, la catena migliore è Qwen3-4B + BM25 + SPLADE + Qwen3-Reranker (0.891, hit@3 1.00); con BGE-M3
   sparse al posto di SPLADE 0.878 restando multilingue.
4. bge-reranker-v2-m3 peggiora sempre (debole sui chunk tabellari); Qwen3-Reranker non ha il difetto ma costa ~1 s/query.
5. I rami sparse da soli sono deboli ma sui codici portano l'MRR a 1.000 nelle catene ibride con filtro.

**Ultimo test richiesto.** "Qwen3-4B + BM25 + SPLADE + RRF" con filtro Qdrant: MRR 0.944, hit@1 0.90,
hit@3 0.98, hit@5 1.00, codici 1.000, cross-lingua 1.000, tabelle 0.698; 0.15 s/query. Rango ≤ 2 su tutte le
domande tranne tre della tabella dati tecnici (peso 4, capacità serbatoio 3, pressione precarica 2).

---

## Stato del repository a fine sessione

- Comandi: `kb-extract`, `kb-index`, `kb-eval`, `kb-search`, `kb-query`, `kb-hybrid`.
- Estrattori: `docling` (default), `mineru`, `pymupdf4llm`, `langchain`.
- Modelli scaricati in locale: BGE-M3 (+ testa sparse), Qwen3-Embedding-0.6B/4B, multilingual-e5-large-instruct,
  bge-reranker-v2-m3, Qwen3-Reranker-0.6B, SPLADE_PP_en_v1, BM25 fastembed, modelli Docling e MinerU.
- Risultati in `eval/results.jsonl` e `eval/results_hybrid.jsonl`; sintesi nel `README.md`.
- Nessun commit ancora eseguito; `pdf/`, `data/`, `blobs/` esclusi da git.
- Il parse server MinerU può essere fermato con `uv run mineru server stop`.
