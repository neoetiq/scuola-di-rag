# scuola-di-rag — KB aziendale da manuali PDF

Pipeline locale (nessun servizio cloud) per costruire una knowledge base multilingue
da manuali di installazione/uso/manutenzione di stufe e caminetti, e per scegliere
il modello di embedding con un benchmark sui propri documenti.

```
pdf/*.pdf  --kb-extract-->  data/md/<estrattore>/*.md  --chunk-->  Chroma index  --kb-eval-->  hit@k / MRR
```

## Setup

```bash
uv sync
```

Python 3.12+, macOS Apple Silicon (i modelli girano su MPS). Nessuna API key necessaria.

## 1. Estrazione PDF -> Markdown

```bash
uv run kb-extract                      # tutti i PDF in ./pdf, backend docling (default)
uv run kb-extract --backend pymupdf4llm "pdf/manuale.pdf"
uv run kb-extract --force              # ri-esegue il backend ignorando la cache in data/md/raw
```

Output: `data/md/<backend>/<nome>.md` con front matter YAML (`source`, `doc_code`, `language`,
`title`) e marcatori `<!-- page: N -->` fra le pagine, così ogni chunk conserva la pagina di origine.

Backend disponibili (`src/kb/extract.py`):

| backend | velocità | pro | contro |
|---|---|---|---|
| `docling` (IBM, default) | ~1-1.5 min / manuale | ordine di lettura corretto su pagine complesse (menu, due colonne), tabelle con celle unite, figure sostituite da segnaposto | lento, modelli ML (~1 GB) scaricati al primo uso, titoli appiattiti |
| `pymupdf4llm` | ~5 s / manuale | veloce, gerarchia titoli da dimensione font | ordine di lettura sbagliato su pagine con schemi, celle unite spezzate |
| `mineru` | ~1-2 min / manuale | layout + VLM (MinerU2.5 1.2B via llama.cpp/Metal), tabelle HTML con colspan | richiede il server locale (vedi sotto), celle multilinea a volte confuse, icone di avvertenza perse |
| `langchain` | ~2 s / manuale | PyPDFLoader (pypdf), il loader di default di LangChain | testo piatto: niente titoli, tabelle ne' struttura; solo come baseline |

Setup MinerU (una tantum, tutto in locale):

```bash
uv run mineru-kit models download --tier standard   # ~2 GB in ~/.mineru/models
uv run mineru config set parse_server.local.mode managed
uv run mineru telemetry disable
uv run mineru server start                           # avvia il parse server locale (porta 16580)
```

Senza `parse_server.local.mode = managed` il CLI di MinerU 4 rifiuta il parse locale e propone il servizio
remoto mineru.net: non usarlo mai con documenti aziendali (`--remote` resta disattivato nel nostro backend).

Pulizia comune a entrambi (`src/kb/clean.py`): rimozione header/footer ripetuti, icone di
avvertenza del font simbolico mappate a `[!DANGER]` / `[!WARNING]` / `[!INFO]`, sillabazione,
normalizzazione dei livelli dei titoli in base alla numerazione (`1` -> `#`, `1.2` -> `##`, `1.2.3` -> `###`).

## 2. Parent document e chunking

`src/kb/parents.py` ricava dal Markdown la struttura a capitoli e sezioni e costruisce i **parent document**:

- un capitolo intero è un parent se sta in `PARENT_MAX_TOKENS` (2000), altrimenti lo sono le sue sezioni;
- le sezioni sotto `PARENT_MIN_TOKENS` (100) vengono accorpate alla vicina (titolo "2.2, 2.3 A / B");
- i parent ancora troppo grandi (capitoli senza sezioni numerate) sono divisi in parti sui sottotitoli;
- stili di titolazione gestiti: "N TITOLO" / "N.M TITOLO", "N.0 TITOLO", titoli maiuscoli non numerati;
  le voci di legenda numerate ("2 LED") sono scartate perché i capitoli devono essere in sequenza.

Sul corpus attuale: 528 parent, mediana 436 token, massimo 1989, copertura del testo 100%.

`src/kb/chunk.py` genera i chunk **dentro** ciascun parent (1200 caratteri, overlap 150), quindi un chunk non
attraversa mai il confine di una sezione. Ogni chunk porta nei metadata, oltre a `doc_code`, `language`,
`page_start`, `page_end`, `section`:

`parent_id`, `parent_title`, `parent_tokens`, `parent_page_start`, `parent_page_end`, `chapter_no`,
`chapter_title`, `section_no`, `section_title`, `chunk_in_parent`.

I file Markdown non vengono divisi e i parent non si leggono tramite offset: il testo di ogni parent è salvato
in un **docstore con chiave `parent_id`**, la collection Qdrant `parents` (solo payload, senza vettori).
`char_start` / `char_end` nel payload del parent restano come provenienza verso il Markdown.
I chunk di copertina e indice non sono indicizzati (`INDEX_FRONT_MATTER = False`): l'indice elenca tutti i
titoli e attira i rami lessicali senza contenere risposte. Il parent corrispondente resta nel docstore.

## 3. Indici ed embedding

Modelli configurati in `src/kb/config.py` (`hf:` = sentence-transformers locale, `ollama:` = server Ollama):

```bash
uv run kb-index --extractor docling --model bge-m3 --model qwen3-4b
```

## 4. Benchmark

Set di domande in `eval/questions.jsonl` (domanda, manuale atteso, pagine che contengono la risposta;
include domande cross-lingua e su codici di allarme).

```bash
uv run kb-eval --extractor docling --verbose
```

Metriche: hit@1/3/5/10 (la pagina giusta compare tra i primi k chunk) e MRR. I risultati vengono
accodati in `eval/results.jsonl`.

## 5. Test interattivo della qualità di estrazione

```bash
uv run kb-query                                   # Qdrant, BGE-M3 dense+sparse (A1), primi 3 chunk
uv run kb-query --parents                         # stessi risultati aggregati in parent document
uv run kb-query --parents --doc H072521IT0 -k 2   # filtro sul manuale, 2 parent
uv run kb-query --pipeline B1                     # altra catena (A0, A1, A2, B0, B1, B1m, B2, B2m)
uv run kb-query --engine chroma --model qwen3-4b  # vecchi indici Chroma, solo dense
```

Chiede una domanda, mostra i risultati e ripete; `quit` per uscire. Con `--parents` recupera `--chunks`
chunk (default 10), li raggruppa per `parent_id` e mostra i primi `-k` parent con numero di chunk trovati,
miglior rango, pagine, token e testo, più il totale di token di contesto.

## 6. Ricerca singola

```bash
uv run kb-search "quanto pesa la stufa" --extractor docling --model bge-m3 -k 5
```

## Risultati (2026-09-23)

Corpus: 2 manuali (P985 THERMO IT, FWTR00 EN), 44 domande (20 IT, 13 EN, 11 cross-lingua, 4 su codici allarme/messaggi).

Confronto estrattore x modello (i due modelli migliori):

| estrattore | modello | hit@1 | hit@3 | hit@5 | MRR | s/query |
|---|---|---|---|---|---|---|
| docling | Qwen3-Embedding-4B | 0.93 | 0.98 | 0.98 | **0.953** | 0.066 |
| docling | BGE-M3 | 0.91 | 0.95 | 0.98 | **0.940** | 0.017 |
| mineru | Qwen3-Embedding-4B | 0.91 | 0.93 | 0.95 | 0.930 | 0.068 |
| pymupdf4llm | Qwen3-Embedding-4B | 0.86 | 0.93 | 0.98 | 0.903 | 0.053 |
| mineru | BGE-M3 | 0.84 | 0.91 | 0.95 | 0.891 | 0.020 |
| langchain (PyPDFLoader) | Qwen3-Embedding-4B | 0.82 | 0.95 | 0.98 | 0.890 | 0.053 |
| pymupdf4llm | BGE-M3 | 0.84 | 0.91 | 0.93 | 0.889 | 0.016 |
| langchain (PyPDFLoader) | BGE-M3 | 0.70 | 0.86 | 0.91 | 0.787 | 0.016 |

Modelli minori (solo docling / pymupdf4llm): Qwen3-Embedding-0.6B MRR 0.880 / 0.848, multilingual-e5-large-instruct 0.830 / 0.746.

Osservazioni:
- **Docling è il miglior estrattore** con entrambi i modelli. MinerU è vicino con Qwen3-4B ma perde 5 punti con BGE-M3.
- **L'estrattore pesa quanto il modello**: fra PyPDFLoader e Docling ci sono 15 punti di MRR con BGE-M3 e 6 con Qwen3-4B.
- **Qwen3-4B è più robusto al testo "sporco"**: su testo piatto senza struttura (PyPDFLoader) resta a 0.89, BGE-M3 crolla a 0.79.
  Su Markdown ben strutturato i due sono quasi equivalenti e BGE-M3 costa 4x meno in query.
- e5-large soffre sulle domande cross-lingua e sui codici di allarme (E8, EMPTY HOPPER).
- Errori residui comuni a tutti: valori nella tabella dati tecnici ("quanti kg di pellet nel serbatoio", "quanto pesa"): il chunk
  giusto c'è ma altre sezioni sullo stesso argomento vincono. Reranker e ricerca ibrida (BGE-M3 sparse) sono il passo successivo.
- Qwen3-Embedding non produce vettori sparse: per la ricerca ibrida con Qwen3 serve un BM25 separato; BGE-M3 li produce nativamente.

## 7. Ricerca ibrida su Qdrant (dense + sparse + RRF + reranker)

`src/kb/hybrid.py` usa Qdrant in modalità embedded (`data/qdrant/<estrattore>/`, nessun server) con due
collection, una per modello dense, ciascuna con tre vettori sparse per chunk:

| vettore | encoder | note |
|---|---|---|
| `dense` | BGE-M3 (coll. `bge`) oppure Qwen3-Embedding-4B (coll. `qwen`) | coseno |
| `bge_sparse` | BGE-M3 sparse (`src/kb/sparse.py`, testa `sparse_linear.pt`) | multilingue, stesso passaggio del dense |
| `bm25` | fastembed `Qdrant/bm25`, senza stemming | indipendente dalla lingua, ideale per i codici |
| `splade` | fastembed `prithivida/Splade_PP_en_v1` | solo inglese |

Le catene sono combinazioni di rami fusi con Reciprocal Rank Fusion dentro Qdrant (`query_points` +
`prefetch`), con reranking opzionale dei primi 20 tramite `BAAI/bge-reranker-v2-m3`.

```bash
uv run kb-hybrid --extractor docling --verbose      # tutte le catene
uv run kb-hybrid --pipeline A2 --pipeline B2m       # solo alcune (prefisso del nome)
```

### Risultati ibrido, corpus 2 manuali (52 domande, di cui 12 su codici)

| catena | hit@1 | hit@3 | MRR | MRR codici | MRR cross | MRR tabelle | s/query |
|---|---|---|---|---|---|---|---|
| A0 BGE-M3 dense | 0.90 | 0.96 | 0.939 | 0.958 | 0.955 | 0.838 | 0.11 |
| A1 BGE-M3 dense + sparse, RRF | 0.88 | 0.94 | 0.921 | **1.000** | 0.841 | 0.807 | 0.07 |
| A2 A1 + reranker | 0.85 | 0.98 | 0.910 | 0.861 | 0.894 | 0.768 | 1.09 |
| B0 Qwen3-4B dense | 0.92 | 0.96 | 0.941 | 0.917 | **1.000** | 0.743 | 0.07 |
| B1 Qwen3-4B + BM25 + SPLADE(en), RRF | 0.83 | 0.92 | 0.888 | **1.000** | 0.758 | 0.668 | 0.08 |
| B2 B1 + reranker | 0.83 | 0.96 | 0.891 | 0.861 | 0.879 | 0.622 | 0.28 |
| B1m Qwen3-4B + BM25 + BGE-M3 sparse, RRF | 0.81 | 0.96 | 0.879 | **1.000** | 0.788 | 0.633 | 0.08 |
| B2m B1m + reranker | 0.83 | 0.96 | 0.893 | 0.861 | 0.879 | 0.636 | 0.29 |
| solo BGE-M3 sparse | 0.71 | 0.85 | 0.779 | 0.917 | 0.636 | 0.599 | 0.07 |
| solo BM25 | 0.60 | 0.77 | 0.686 | 0.833 | 0.473 | 0.576 | 0.07 |
| solo SPLADE (en) | 0.73 | 0.83 | 0.785 | 0.903 | 0.545 | 0.556 | 0.07 |

Con soli 2 manuali il dense da solo è già al limite e i rami sparse aiutano solo sui codici
(MRR codici 1.000 in tutte le catene ibride) mentre penalizzano cross-lingua e tabelle; il reranker
non aiuta. Il confronto significativo è quello sul corpus completo (sotto).

### Risultati ibrido, corpus completo (13 manuali, 1935 chunk, 52 domande)

Modalità `plain` (domanda così com'è, nessuna indicazione del prodotto):

| catena | hit@1 | hit@3 | hit@5 | MRR | MRR codici | MRR cross | MRR tabelle | s/query |
|---|---|---|---|---|---|---|---|---|
| A0 BGE-M3 dense | 0.52 | 0.67 | 0.75 | 0.620 | 0.843 | 0.710 | 0.330 | 0.07 |
| A1 BGE-M3 dense + sparse, RRF | 0.54 | 0.67 | 0.75 | 0.628 | 0.903 | 0.658 | 0.391 | 0.09 |
| A2 A1 + reranker | 0.48 | 0.63 | 0.77 | 0.596 | 0.875 | 0.579 | 0.272 | 0.40 |
| B0 Qwen3-4B dense | **0.62** | **0.73** | 0.79 | **0.706** | 0.875 | **0.909** | 0.439 | 0.07 |
| B1 Qwen3-4B + BM25 + SPLADE(en), RRF | 0.56 | 0.63 | 0.75 | 0.640 | 0.938 | 0.652 | 0.428 | 0.11 |
| B2 B1 + reranker | 0.48 | 0.71 | **0.88** | 0.623 | 0.861 | 0.609 | 0.335 | 0.36 |
| B1m Qwen3-4B + BM25 + BGE-M3 sparse, RRF | 0.60 | 0.67 | 0.73 | 0.674 | **0.958** | 0.772 | **0.452** | 0.10 |
| B2m B1m + reranker | 0.48 | 0.71 | 0.87 | 0.621 | 0.861 | 0.609 | 0.326 | 0.36 |
| solo BGE-M3 sparse | 0.38 | 0.58 | 0.62 | 0.509 | 0.833 | 0.409 | 0.299 | 0.09 |
| solo BM25 | 0.37 | 0.52 | 0.62 | 0.471 | 0.697 | 0.402 | 0.229 | 0.08 |
| solo SPLADE (en) | 0.40 | 0.52 | 0.65 | 0.497 | 0.646 | 0.205 | 0.275 | 0.09 |

Attenzione: i 9 manuali italiani condividono capitoli identici (presa d'aria, pulizia, manutenzione, distanze),
quindi molte domande in italiano senza indicazione del prodotto sono ambigue per costruzione e il punteggio
stretto le conta come errori. Le domande inglesi (manuale unico) restano quasi tutte a rango 1.
Le modalità `product` (nome del prodotto nella domanda) e `filter` (filtro Qdrant sul manuale) sotto
rimuovono l'ambiguità e sono la misura corretta della qualità delle catene.

Modalità `product` (nome del prodotto nella domanda, es. "... (P985 Thermo)") e `filter` (filtro Qdrant sul manuale):

| catena | MRR product | hit@3 product | MRR tabelle product | MRR filter | hit@1 filter | MRR codici filter | s/query |
|---|---|---|---|---|---|---|---|
| A0 BGE-M3 dense | 0.864 | 0.94 | 0.698 | **0.975** | **0.96** | 1.000 | 0.08 |
| A1 BGE-M3 dense + sparse, RRF | 0.831 | 0.94 | 0.615 | 0.969 | 0.96 | 1.000 | 0.10 |
| A2 A1 + bge-reranker-v2-m3 | 0.839 | 0.88 | 0.802 | 0.903 | 0.83 | 0.861 | 0.42 |
| A2 A1 + Qwen3-Reranker-0.6B | 0.870 | 0.98 | 0.698 | 0.913 | 0.85 | 0.958 | 1.08 |
| B0 Qwen3-4B dense | 0.857 | 0.94 | **0.854** | 0.948 | 0.92 | 0.933 | 0.07 |
| B1 Qwen3-4B + BM25 + SPLADE(en), RRF | 0.756 | 0.88 | 0.708 | 0.944 | 0.90 | 1.000 | 0.14 |
| B2 B1 + bge-reranker-v2-m3 | 0.854 | 0.94 | 0.812 | 0.885 | 0.81 | 0.861 | 0.38 |
| B2 B1 + Qwen3-Reranker-0.6B | **0.891** | **1.00** | 0.771 | 0.908 | 0.85 | 0.958 | 1.27 |
| B1m Qwen3-4B + BM25 + BGE-M3 sparse, RRF | 0.774 | 0.92 | 0.698 | 0.897 | 0.85 | 1.000 | 0.13 |
| B2m B1m + bge-reranker-v2-m3 | 0.853 | 0.94 | 0.810 | 0.884 | 0.81 | 0.861 | 0.37 |
| B2m B1m + Qwen3-Reranker-0.6B | 0.878 | 1.00 | 0.750 | 0.908 | 0.85 | 0.958 | 1.32 |

Conclusioni:
- **Il filtro sul manuale è la leva più forte**: da 0.86 a 0.975 di MRR. L'applicazione deve far scegliere il
  prodotto o riconoscerlo dalla domanda, prima di qualunque raffinamento del retrieval.
- **Con il filtro, il dense da solo (BGE-M3) è imbattibile** e sparse/reranker non aggiungono nulla.
- **Senza filtro, la catena migliore è Qwen3-4B + BM25 + SPLADE + Qwen3-Reranker** (MRR 0.891, hit@3 1.00):
  i rami sparse allargano i candidati, il reranker sceglie. Con BGE-M3 sparse al posto di SPLADE si ottiene
  quasi lo stesso (0.878) restando multilingue.
- **bge-reranker-v2-m3 peggiora sempre**: sottovaluta i chunk tabellari e preferisce i paragrafi che
  "parlano" dell'argomento alla riga con il valore. Qwen3-Reranker non ha questo difetto ma costa ~1 s/query.
- I rami sparse da soli sono deboli (MRR 0.47-0.54) ma sui codici fanno la differenza: BM25 e BGE-M3 sparse
  portano l'MRR sui codici a 1.000 nelle catene ibride con filtro.
- Qwen3-Embedding non ha vettori sparse: nella catena B il lessicale è BM25 (fastembed, senza stemming).

### Dopo l'introduzione dei parent document (1905 chunk, 528 parent)

Il chunking ora rispetta i confini di sezione e non indicizza copertina e indice. Controllo di regressione (MRR):

| catena | filter prima | filter dopo | product prima | product dopo |
|---|---|---|---|---|
| A0 BGE-M3 dense | 0.975 | 0.966 | 0.864 | 0.884 |
| A1 BGE-M3 dense + sparse, RRF | 0.969 | 0.974 | 0.831 | 0.896 |
| B0 Qwen3-4B dense | 0.948 | 0.939 | 0.857 | 0.860 |
| B1 Qwen3-4B + BM25 + SPLADE, RRF | 0.944 | 0.907 | 0.756 | 0.709 |

Le catene BGE-M3 migliorano o restano uguali, Qwen dense è invariato, la catena con BM25 + SPLADE peggiora:
ora che anche P961 e Fulvia hanno capitoli riconosciuti, i loro chunk competono di più sui rami lessicali.

L'ingestione carica un modello alla volta (sparse ONNX, poi BGE-M3, poi Qwen3-4B con batch 4) e rilascia la
memoria Metal tra le fasi: tenendoli tutti in memoria il processo veniva terminato dal sistema (exit 137).
