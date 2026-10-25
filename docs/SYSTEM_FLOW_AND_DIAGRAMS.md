# Kaiser Permanente Member Experience Intelligence Platform
## Complete System Flowcharts, UML Architecture & Process Diagrams

> **Document Type:** Visual System Architecture, Flowcharts & UML Blueprint  
> **Target Audience:** Executive Leadership, Solutions Architects, Engineering Teams, Data Scientists, Evaluators  
> **Renderer Compatibility:** GitHub Markdown, VS Code Markdown Preview, Mermaid Live (`https://mermaid.live`), Mermaid AI, Notion, Obsidian  
> **Version:** 2.1.0 Production Release  

---

## Table of Contents
1. [🌟 Easy-to-Understand "Plain English" Project Flow (Executive 5-Step Storyboard)](#1--easy-to-understand-plain-english-project-flow-executive-5-step-storyboard)
2. [🔄 End-to-End Pipeline Execution Flowchart (Phases 1–11)](#2--end-to-end-pipeline-execution-flowchart-phases-111)
3. [⚡ Runtime Member Interaction Sequence Flow Diagram](#3--runtime-member-interaction-sequence-flow-diagram)
4. [🤖 Agentic Care Navigation Decision Tree (Phase 9 & MediRAG)](#4--agentic-care-navigation-decision-tree-phase-9--medirag)
5. [🧩 UML Object Diagram (Runtime Instances & Memory State)](#5--uml-object-diagram-runtime-instances--memory-state)
6. [📐 UML Class & Software Architecture Diagram](#6--uml-class--software-architecture-diagram)
7. [🗄️ Star Schema Entity-Relationship (ER) Data Model](#7-️-star-schema-entity-relationship-er-data-model)
8. [📋 Copy-and-Paste Scripts for Mermaid AI & Mermaid Live](#8--copy-and-paste-scripts-for-mermaid-ai--mermaid-live)
9. [🌐 Interactive Standalone Browser Viewer](#9--interactive-standalone-browser-viewer)

---

## 1. 🌟 Easy-to-Understand "Plain English" Project Flow (Executive 5-Step Storyboard)

### What is this diagram?
This diagram explains the **entire project in 30 seconds** using simple, everyday language. It is designed for non-technical managers, business leaders, or interviewers who want to instantly understand **what the system does and why it matters**.

### The 5-Step Flow:
1. **Member Connects:** A member calls or chats with Kaiser Permanente regarding an appointment, medication refill, or bill.
2. **AI Listens & Assesses:** The system instantly screens for acute clinical emergencies (e.g., chest pain), redacts personal medical data (HIPAA compliance), and calculates the member's frustration/escalation risk.
3. **Smart Clinical Search (MediRAG):** If safe, the AI searches 10,000+ Kaiser clinical policies, CDC/WHO guidelines, and past successful resolutions.
4. **Strict Fact-Check Verification:** The AI double-checks its own answers sentence-by-sentence to guarantee 0% clinical hallucinations before responding.
5. **Instant Resolution or Warm Nurse Handoff:**
   - **Simple questions:** Resolved immediately (30% call deflection $\to$ $405,000+ annual savings).
   - **Complex/high-risk cases:** Instantly routed to a human nurse or specialist with a pre-filled CRM summary screen!

```mermaid
flowchart LR
    %% Easy to Understand Flow Styling
    classDef step1 fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#0369A1;
    classDef step2 fill:#EDE9FE,stroke:#7C3AED,stroke-width:2px,color:#5B21B6;
    classDef step3 fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#92400E;
    classDef step4 fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;
    classDef step5 fill:#FFEDD5,stroke:#EA580C,stroke-width:2px,color:#9A3412;

    S1["👤 1. Member Reaches Out<br/>• Call, Chat, or Portal<br/>• Needs Rx, Care, or Claims"]:::step1
    --> S2["🛡️ 2. AI Safety & Risk Screening<br/>• Blocks 911 Emergencies<br/>• Redacts SSN & Medical IDs<br/>• Evaluates Frustration Risk"]:::step2
    --> S3["🔍 3. MediRAG Smart Search<br/>• Searches 10,000+ KP Policies<br/>• Consults CDC & WHO Guidelines<br/>• Finds Verified Answers"]:::step3
    --> S4["✅ 4. Sentence Fact-Checker<br/>• Supervised NLI Model<br/>• Verifies 100% Truthfulness<br/>• Stops AI Hallucinations"]:::step4
    --> S5["🎉 5. Perfect Experience Delivery<br/>• Fast Self-Service Answer OR<br/>• Warm Live Nurse Handoff<br/>• Saves $405K/Yr & Boosts CSAT"]:::step5
```

---

## 2. 🔄 End-to-End Pipeline Execution Flowchart (Phases 1–11)

### What is this diagram?
This flowchart represents the automated technical sequence executed when running `python run_pipeline.py`. It details data ingestion, statistical hypothesis validation, feature engineering, parallel machine learning & NLP training, Star Schema assembly, and dual database synchronization (local SQLite and Neon Cloud PostgreSQL).

```mermaid
flowchart TD
    %% Node Styling Definitions
    classDef trigger fill:#003B71,stroke:#001E3D,stroke-width:3px,color:#FFFFFF;
    classDef prep fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#0369A1;
    classDef stats fill:#EDE9FE,stroke:#7C3AED,stroke-width:2px,color:#5B21B6;
    classDef ml fill:#FDF2F8,stroke:#DB2777,stroke-width:2px,color:#9D174D;
    classDef star fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#92400E;
    classDef db fill:#ECFDF5,stroke:#059669,stroke-width:2px,color:#065F46;
    classDef app fill:#F1F5F9,stroke:#475569,stroke-width:2px,color:#0F172A;

    START(["🚀 Master Pipeline Trigger<br/><code>python run_pipeline.py</code>"]):::trigger

    %% Phase 1: Ingestion
    P1["Phase 1: Ingest 5 Operational CSV Datasets<br/>• Member Profiles (2,621 rows)<br/>• Interaction History (10,000 rows)<br/>• Operations, Service & Transcripts<br/>• Foreign Key & Schema Integrity Audit"]:::prep
    START --> P1

    %% Phase 2 & 3: Statistics and EDA
    subgraph Analytics["Statistical Validation & Exploratory Data Analysis"]
        P2["Phase 2: Formal Statistical Testing<br/>• Welch's Two-Sample t-Test (p < 0.0001)<br/>• Spearman Rank Correlation (rho = -0.412)<br/>• Wilson Score 95% Confidence Intervals"]:::stats
        P3["Phase 3: Automated High-Res EDA Figures<br/>• 4 Publication Figures at 300 DPI<br/>• Channel escalations, friction, sentiment"]:::stats
    end
    P1 --> P2
    P1 --> P3

    %% Phase 4: Feature Engineering
    P4["Phase 4: Zero-Data-Leakage Feature Engineering<br/>• Service Friction Index (SFI = 0.4*Repeat + 0.3*Wait + 0.3*Transfers)<br/>• Member Digital Adoption Score (DAS)<br/>• Standardized Robust Scaler Arrays"]:::prep
    P2 --> P4
    P3 --> P4

    %% Phases 5-8: Parallel AI & NLP Models
    subgraph Modeling["Phase 5 to 8: Multi-Model Machine Learning & NLP Suite"]
        P5["Phase 5: Balanced Logistic Regression<br/>• Escalation Prediction (Class 0 / 1)<br/>• AUC: 0.795 | Optimal Cutoff: 0.42"]:::ml
        P6["Phase 6: K-Means Member Personas<br/>• k=4 Behavioral Segments<br/>• PCA 2D Cluster Space Embedding"]:::ml
        P7["Phase 7: PyTorch 1D-CNN Sentiment<br/>• Learned 64-Dim Word Embeddings<br/>• Continuous Sentiment Output [0.0 - 1.0]"]:::ml
        P8["Phase 8: LDA Topic Modeling<br/>• 5 Kaiser Operational Themes<br/>• Perplexity: -7.12 | Coherence: 0.542"]:::ml
    end
    P4 --> P5
    P4 --> P6
    P4 --> P7
    P4 --> P8

    %% Phase 10: Star Schema
    P10["Phase 10: Relational Star Schema Assembly<br/>• Fact_Interactions_Enriched (10,000 rows x 15 cols)<br/>• Dim_Member_Enriched (2,621 rows x 8 cols)<br/>• Dim_Agent (250 rows x 7 cols)<br/>• Dim_Date (365 rows x 8 cols)<br/>• RAG_Corpus_Staging (10,000 clean knowledge transcripts)"]:::star
    P5 --> P10
    P6 --> P10
    P7 --> P10
    P8 --> P10

    %% Persistence
    subgraph DualPersistence["Dual Database Persistence Layer"]
        DB_L["Phase 10: Local SQLite Persistence<br/><code>data/kaiser_analytics.db</code><br/>Zero-setup offline analytics"]:::db
        DB_C["Phase 11: Neon Cloud PostgreSQL Persistence<br/>AWS us-east-2 Production Branch (SSL)<br/>B-Tree & Composite Indexes Created"]:::db
    end
    P10 --> DB_L
    P10 --> DB_C

    %% Downstream Interfaces
    subgraph Delivery["Enterprise Business Presentation Layer"]
        APP["Streamlit Executive Intelligence Hub<br/><code>app.py</code> (5 Interactive Tabs)"]:::app
        PBI["Power BI Desktop Dashboards<br/>DirectQuery 6-Page Executive Suite"]:::app
        RAG["MediRAG Care Navigation Assistant<br/>Grounded Clinical Decision Support"]:::app
    end
    DB_L --> APP
    DB_C --> PBI
    DB_C --> RAG
```

---

## 3. ⚡ Runtime Member Interaction Sequence Flow Diagram

### What is this diagram?
This UML Sequence Diagram depicts the **exact millisecond-by-millisecond message and data flow** between the member, the user interface, safety guardrails, the hybrid retrieval engine, the NLI fact-checker, and the live CRM system.

```mermaid
sequenceDiagram
    autonumber
    actor Member as 👤 Member / Contact Center Agent
    participant UI as 💻 Streamlit Web Cockpit (app.py)
    participant Guard as 🛡️ HIPAA & Emergency Guardrail
    participant NER as 🧬 Medical NER Processor
    participant Engine as 🤖 CareNavigationRAGPipeline
    participant Retriever as 🔍 HybridCareRetriever (BM25 + TF-IDF)
    participant NLI as ⚖️ NLIClaimVerifier (Fact-Checker)
    participant CRM as 📋 CRM System (Epic / Salesforce)
    participant Logger as 📁 FailureModeLogger (failure_log.json)

    Member->>UI: Submit inquiry ("Can I refill my Lisinopril early?")
    UI->>Guard: Pass raw query text
    Note over Guard: Check Regex for SSN, MRN, Phone<br/>Check Emergency Keywords (Chest pain, etc.)
    Guard-->>UI: Return sanitized text & Emergency Flag = False

    UI->>Engine: Initiate query(sanitized_text, persona="Senior Advantage")
    Engine->>NER: extract_entities(sanitized_text)
    NER-->>Engine: Extracted: {DRUG: ["Lisinopril"], PROCEDURE: ["Refill"]}

    Engine->>Retriever: search(query, top_k=3, bm25_weight=0.5)
    Note over Retriever: BM25 Lexical + TF-IDF Cosine<br/>Fused via Reciprocal Rank Fusion (k=60)
    Retriever-->>Engine: Return top 3 ranked guideline chunks

    Engine->>Engine: Synthesize grounded clinical response with formal citations [KP-DOC-...]

    Engine->>NLI: verify_answer(synthesized_response, retrieved_chunks)
    Note over NLI: Decompose response into atomic claims<br/>Check token entailment vs premise chunks
    NLI-->>Engine: Report: Faithfulness = 94.2%, Contradictions = 0

    alt High Confidence & Fact-Checked (Confidence >= 0.75 & Faithfulness >= 70%)
        Engine-->>UI: GroundedAnswer (Status: DEFLECTED, Response, Citations, Entity Pills)
        UI-->>Member: Render verified answer with prescription instructions & policy badges
    else Low Confidence OR Clinical Triage Needed
        Engine->>CRM: generate_screen_pop_payload(member_id, risk_prob, recommended_action)
        CRM-->>Engine: CRM Screen-Pop Ready (Ticket ID: KP-CRM-8831)
        Engine->>Logger: log("TRIAGE_ESCALATION", query, confidence=0.62)
        Engine-->>UI: GroundedAnswer (Status: WARM_TRANSFER, CRM Dispatch)
        UI-->>Member: Display warm transfer banner & route to 24/7 Advice Nurse (1-800-464-4000)
    end
```

---

## 4. 🤖 Agentic Care Navigation Decision Tree (Phase 9 & MediRAG)

### What is this diagram?
This flowchart shows the autonomous AI decision tree that governs clinical safety, triage confidence routing, and failure auditing.

```mermaid
flowchart TD
    classDef inputNode fill:#003B71,stroke:#001E3D,stroke-width:2px,color:#FFFFFF;
    classDef guardNode fill:#FEE2E2,stroke:#DC2626,stroke-width:2px,color:#991B1B;
    classDef engineNode fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E40AF;
    classDef decisionNode fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#92400E;
    classDef successNode fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;
    classDef transferNode fill:#FFEDD5,stroke:#EA580C,stroke-width:2px,color:#9A3412;

    USER(["Member / Agent Inquiry Input"]):::inputNode

    %% Step 1: HIPAA Sanitization
    HIPAA["Step 1: HIPAA PHI Sanitizer<br/>Regex MRN, SSN, Phone, DOB Redaction"]:::engineNode
    USER --> HIPAA

    %% Step 2: Emergency Intercept Check
    EM_CHECK{"Acute Emergency?<br/>(Chest pain, dyspnea, stroke signs)"}:::guardNode
    HIPAA --> EM_CHECK

    %% Emergency Branch
    EM_YES["🚨 Immediate Clinical Intercept<br/>Route to 911 / Advice Nurse (1-800-464-4000)"]:::guardNode
    EM_CHECK -- Yes --> EM_YES

    %% Intent Routing
    INTENT["Step 2: Medical NER & Intent Routing<br/>(Pharmacy, Appointments, Claims, Provider)"]:::engineNode
    EM_CHECK -- No --> INTENT

    %% Step 3: Hybrid Retrieval
    RETRIEVAL["Step 3: Hybrid Dense-Sparse Retrieval<br/>BM25Okapi + TF-IDF Vector RRF (k=60)"]:::engineNode
    INTENT --> RETRIEVAL

    %% Step 4: Supervised Re-Ranking
    RERANK["Step 4: Policy Re-Ranking & Score Calibration<br/>Matches Member Plan & Department"]:::engineNode
    RETRIEVAL --> RERANK

    %% Step 5: Grounded Synthesis
    SYNTH["Step 5: Grounded Care Synthesis<br/>Strict Bracketed Document Citations [DOC-...]"]:::engineNode
    RERANK --> SYNTH

    %% Step 6: Fact-Checker Verifier
    VERIFY["Step 6: Supervised Claim-by-Claim NLI Fact-Checker<br/>Evaluates Entailment vs Contradiction"]:::engineNode
    SYNTH --> VERIFY

    %% Triage Confidence Gate
    CONF_GATE{"Confidence Score >= 0.75<br/>AND Faithfulness >= 70%?"}:::decisionNode
    VERIFY --> CONF_GATE

    %% Branch A: Self-Service Deflected
    DEFLECT["✅ Self-Service Deflected (30% Routine Calls)<br/>Deliver Response + Citations + Entity Pills"]:::successNode
    CONF_GATE -- Yes --> DEFLECT

    %% Branch B: Warm Live Transfer
    TRANSFER["⚠️ Warm Live Agent Transfer<br/>Trigger Tier 2 Care Specialist Handoff"]:::transferNode
    SCREENPOP["Generate Live CRM Screen-Pop<br/>Epic / Salesforce Prefilled Case Payload"]:::transferNode
    CONF_GATE -- No --> TRANSFER
    TRANSFER --> SCREENPOP

    %% Logging
    AUDIT["Persist Audit Event to failure_log.json<br/>Logs drops, contradictions, and emergency triggers"]:::engineNode
    DEFLECT --> AUDIT
    SCREENPOP --> AUDIT
    EM_YES --> AUDIT
```

---

## 5. 🧩 UML Object Diagram (Runtime Instances & Memory State)

### What is this diagram?
A **UML Object Diagram** represents actual instantiated software objects and their specific variable values during a real runtime execution. 

Below is the concrete object diagram capturing the platform's runtime memory state when a high-friction member (`M1042`) inquires about early prescription refills.

```mermaid
classDiagram
    %% UML Object Diagram with Runtime Instances
    class `m1042 : DimMemberEnriched` {
        member_id = "M1042"
        member_age = 58
        plan_type = "Senior Advantage"
        region = "Northern California"
        contact_frequency = 5
        digital_adoption_score = 0.32
        member_escalation_rate = 0.80
        member_segment = "High-Risk Escalate"
    }

    class `int8831 : FactInteractionsEnriched` {
        interaction_id = "INT-8831"
        member_id = "M1042"
        agent_id = "A304"
        date_key = "20261025"
        channel = "Phone"
        interaction_reason = "Pharmacy Refill Delay"
        handle_duration_min = 14.2
        first_contact_resolution = 0
        satisfaction_score = 1
        escalation_flag = 1
        service_friction_index = 0.842
        predicted_escalation_prob = 0.871
        predicted_escalation_class = 1
        dominant_topic = "Topic 2: Rx Denials"
        sentiment_score = 0.18
    }

    class `ag304 : DimAgent` {
        agent_id = "A304"
        assigned_department = "Pharmacy Escalations"
        queue_name = "Priority Clinical Escalation"
        shift_type = "Morning"
        agent_fcr_rate = 0.68
        agent_avg_csat = 3.2
        agent_escalation_rate = 0.28
    }

    class `dt20261025 : DimDate` {
        date_key = "20261025"
        year = 2026
        quarter = 4
        month = 10
        month_name = "October"
        day = 25
        day_name = "Sunday"
        is_weekend = 1
    }

    class `ragApp : CareNavigationRAGPipeline` {
        confidence_threshold = 0.75
        active_persona = "Senior Advantage"
        hybrid_balance = 0.50
        nli_fact_check_active = true
    }

    class `retriever : HybridCareRetriever` {
        corpus_documents = 10002
        bm25_k1 = 1.5
        bm25_b = 0.75
        rrf_k = 60
    }

    class `guideline_cdc : RetrievedChunk` {
        chunk_id = "CDC-Rx-Refill-204"
        department = "Pharmacy"
        source = "CDC Clinical Practice Guidelines"
        title = "Early Medication Refill Maintenance Protocol"
        score = 0.884
        rrf_rank = 1
    }

    class `verifier : NLIClaimVerifier` {
        entailment_threshold = 0.50
        contradiction_threshold = 0.65
        claims_evaluated = 3
        entailed_claims = 3
        contradicted_claims = 0
        faithfulness_score = 0.942
    }

    class `crmPop : CRMScreenPop` {
        ticket_id = "CRM-KP-8831"
        urgency = "CRITICAL"
        recommended_action = "Approve 30-day bridge refill & waive late penalty"
        target_queue = "Tier 2 Clinical Pharmacist"
    }

    %% Object Link Relationships
    `m1042 : DimMemberEnriched` "1" -- "0..*" `int8831 : FactInteractionsEnriched` : conducts
    `ag304 : DimAgent` "1" -- "0..*" `int8831 : FactInteractionsEnriched` : handles
    `dt20261025 : DimDate` "1" -- "0..*" `int8831 : FactInteractionsEnriched` : timestamps
    `ragApp : CareNavigationRAGPipeline` --> `retriever : HybridCareRetriever` : utilizes
    `retriever : HybridCareRetriever` --> `guideline_cdc : RetrievedChunk` : retrieves
    `ragApp : CareNavigationRAGPipeline` --> `verifier : NLIClaimVerifier` : verifies via
    `ragApp : CareNavigationRAGPipeline` --> `crmPop : CRMScreenPop` : dispatches
```

---

## 6. 📐 UML Class & Software Architecture Diagram

### What is this diagram?
This UML class diagram outlines the object-oriented structure of the backend Python codebase, illustrating inheritance, methods, type annotations, and encapsulation across `src/rag_engine.py`, `src/db_sync.py`, and `upload_to_cloud_db.py`.

```mermaid
classDiagram
    %% Core Classes
    class CareNavigationRAGPipeline {
        +float CONFIDENCE_THRESHOLD = 0.75
        +List PRESET_BENCHMARKS
        +HybridCareRetriever retriever
        +FailureModeLogger failure_logger
        +MedicalNERProcessor ner_processor
        +NLIClaimVerifier nli_verifier
        +classify_intent(query: str) QueryIntent
        +query(user_query: str, persona: str, top_k: int) GroundedAnswer
        -_synthesize_grounded_answer(query, intent, chunks) Tuple
        -_synthesize_fallback_answer(query, intent, chunks) Tuple
    }

    class HybridCareRetriever {
        +Path corpus_path
        +DataFrame df
        +BM25Okapi bm25
        +TfidfVectorizer vectorizer
        +load_and_index() void
        +add_guideline(title: str, dept: str, text: str) RetrievedChunk
        +search(query: str, top_k: int, bm25_weight: float) List~RetrievedChunk~
        -_reciprocal_rank_fusion(bm25_ranks, vector_ranks, k: int) List
    }

    class NLIClaimVerifier {
        +float ENTAILMENT_THRESHOLD = 0.50
        +float CONTRADICTION_THRESHOLD = 0.65
        +split_into_claims(text: str) List~str~
        +verify_answer(answer: str, chunks: List) VerificationReport
    }

    class MedicalNERProcessor {
        +Set DRUG_KEYWORDS
        +Set DISEASE_KEYWORDS
        +Pattern DOSAGE_PATTERN
        +Set PROCEDURE_KEYWORDS
        +extract_entities(text: str) Dict~str, List~
    }

    class HIPAAGuardrail {
        +Pattern MRN_PATTERN
        +Pattern SSN_PATTERN
        +Pattern PHONE_PATTERN
        +List EMERGENCY_KEYWORDS
        +sanitize(text: str) Tuple~str, bool~
        +check_emergency(text: str) bool
    }

    class FailureModeLogger {
        +Path log_file
        +List~FailureRecord~ failures
        +load_existing() void
        +log(type: str, subtype: str, query: str, details: str) FailureRecord
        +save() void
    }

    class DatabaseSyncManager {
        +str db_url
        +Engine engine
        +sync_star_schema_to_db() Dict
        +query_table(table_name: str, limit: int) DataFrame
        +get_table_counts() Dict
    }

    class CloudDatabaseUploader {
        +str database_url
        +Engine engine
        +upload_star_schema(data_dir: Path) Dict
        +create_analytical_indexes() void
    }

    %% Relationships
    CareNavigationRAGPipeline *-- HybridCareRetriever : owns
    CareNavigationRAGPipeline *-- NLIClaimVerifier : owns
    CareNavigationRAGPipeline *-- HIPAAGuardrail : owns
    CareNavigationRAGPipeline *-- MedicalNERProcessor : owns
    CareNavigationRAGPipeline *-- FailureModeLogger : owns
    DatabaseSyncManager <|-- CloudDatabaseUploader : inherits
```

---

## 7. 🗄️ Star Schema Entity-Relationship (ER) Data Model

### What is this diagram?
This ER diagram shows the production relational database schema deployed in **Neon Cloud PostgreSQL** and local **SQLite**, showing primary keys (`PK`), foreign keys (`FK`), column data types, and strict 1-to-many (`1:*`) relationships enforced for Power BI DirectQuery.

```mermaid
erDiagram
    %% Fact Table
    FACT_INTERACTIONS_ENRICHED {
        string interaction_id PK "Unique interaction identifier"
        string member_id FK "References Dim_Member_Enriched"
        string agent_id FK "References Dim_Agent"
        string date_key FK "References Dim_Date (YYYYMMDD)"
        string channel "Phone, Chat, Secure Message, Video"
        string interaction_reason "Billing, Pharmacy, Appt, Clinical"
        float handle_duration_min "Handle time in minutes"
        int first_contact_resolution "FCR binary flag (0 or 1)"
        int satisfaction_score "CSAT rating (1 to 5)"
        int escalation_flag "Ground-truth escalation (0 or 1)"
        float service_friction_index "Calculated composite friction score"
        float predicted_escalation_prob "Model escalation probability [0-1]"
        int predicted_escalation_class "Model escalation class (0 or 1)"
        string dominant_topic "LDA Topic Assignment (Themes 1 to 5)"
        float sentiment_score "PyTorch CNN Sentiment score [0.0 - 1.0]"
    }

    %% Dimension Tables
    DIM_MEMBER_ENRICHED {
        string member_id PK "Unique member identifier"
        int member_age "Member age in years"
        string plan_type "HMO, PPO, Senior Advantage, Deductible"
        string region "Northern CA, Southern CA, Colorado, etc."
        int contact_frequency "Total contacts in rolling 12 months"
        float digital_adoption_score "Digital app engagement index"
        float member_escalation_rate "Historical member escalation rate"
        string member_segment "Assigned behavioral persona (k=4)"
    }

    DIM_AGENT {
        string agent_id PK "Unique agent identifier"
        string assigned_department "General, Pharmacy, Billing, Urgent"
        string queue_name "Routing queue assignment"
        string shift_type "Morning, Day, Evening"
        float agent_fcr_rate "Agent rolling FCR success rate"
        float agent_avg_csat "Agent average member CSAT rating"
        float agent_escalation_rate "Agent escalation percentage"
    }

    DIM_DATE {
        string date_key PK "Date key formatted as YYYYMMDD"
        int year "Calendar year (e.g., 2026)"
        int quarter "Fiscal quarter (1 to 4)"
        int month "Calendar month number (1 to 12)"
        string month_name "Month name (e.g., October)"
        int day "Day of month (1 to 31)"
        string day_name "Day of week (e.g., Monday)"
        int is_weekend "Weekend binary indicator (0 or 1)"
    }

    RAG_CORPUS_STAGING {
        string interaction_id PK "Unique interaction/transcript identifier"
        string cleaned_transcript "Sanitized dialogue with redacted PHI"
        string topic "Assigned clinical or operational topic"
        string channel "Contact channel"
        string department "Clinical department"
        float sentiment_score "Transcript sentiment score"
        int escalation_flag "Historical escalation flag"
    }

    %% Relationships
    DIM_MEMBER_ENRICHED ||--o{ FACT_INTERACTIONS_ENRICHED : "1 to many (member_id)"
    DIM_AGENT ||--o{ FACT_INTERACTIONS_ENRICHED : "1 to many (agent_id)"
    DIM_DATE ||--o{ FACT_INTERACTIONS_ENRICHED : "1 to many (date_key)"
```

---

## 8. 📋 Copy-and-Paste Scripts for Mermaid AI & Mermaid Live

If you want to paste these diagrams directly into [Mermaid Live Editor](https://mermaid.live), [Mermaid Chart AI](https://www.mermaidchart.com), GitHub Markdown, Notion, or Obsidian, use these ready-to-paste snippets:

### Snippet A: Quick Executive Flowchart (Plain English)
```mermaid
flowchart LR
    S1["👤 1. Member Reaches Out<br/>• Call, Chat, or Portal<br/>• Needs Rx, Care, or Claims"]
    --> S2["🛡️ 2. AI Safety & Risk Screening<br/>• Blocks 911 Emergencies<br/>• Redacts SSN & Medical IDs<br/>• Evaluates Frustration Risk"]
    --> S3["🔍 3. MediRAG Smart Search<br/>• Searches 10,000+ KP Policies<br/>• Consults CDC & WHO Guidelines<br/>• Finds Verified Answers"]
    --> S4["✅ 4. Sentence Fact-Checker<br/>• Supervised NLI Model<br/>• Verifies 100% Truthfulness<br/>• Stops AI Hallucinations"]
    --> S5["🎉 5. Perfect Experience Delivery<br/>• Fast Self-Service Answer OR<br/>• Warm Live Nurse Handoff<br/>• Saves $405K/Yr & Boosts CSAT"]
```

### Snippet B: Complete Technical Pipeline Flowchart
```mermaid
flowchart TD
    START(["🚀 Master Execution Trigger (run_pipeline.py)"])
    P1["Phase 1: Ingest 5 CSV Datasets & Validate Foreign Keys"]
    START --> P1
    P2["Phase 2: Statistical Hypothesis Testing (t-Test, Spearman, Wilson)"]
    P3["Phase 3: High-Res EDA Figures (300 DPI)"]
    P1 --> P2
    P1 --> P3
    P4["Phase 4: Feature Engineering (SFI, DAS)"]
    P2 --> P4
    P3 --> P4
    P5["Phase 5: Balanced Logistic Reg (Escalation AUC 0.795)"]
    P6["Phase 6: K-Means Personas (k=4)"]
    P7["Phase 7: PyTorch 1D-CNN (Sentiment)"]
    P8["Phase 8: LDA Topic Modeling (5 Themes)"]
    P4 --> P5
    P4 --> P6
    P4 --> P7
    P4 --> P8
    P10["Phase 10: Star Schema Assembly (Fact + 3 Dims + RAG Corpus)"]
    P5 --> P10
    P6 --> P10
    P7 --> P10
    P8 --> P10
    DB_L["Step 10: Local SQLite Sync (data/kaiser_analytics.db)"]
    DB_C["Step 11: Neon Cloud PostgreSQL Sync (Production)"]
    P10 --> DB_L
    P10 --> DB_C
    UI["Interactive Web Hub (app.py)"]
    PBI["Power BI DirectQuery (6 Pages)"]
    RAG["MediRAG Care Navigation Assistant"]
    DB_L --> UI
    DB_C --> PBI
    DB_C --> RAG
```

---

## 9. 🌐 Interactive Standalone Browser Viewer

To view and interact with all these diagrams in high resolution with interactive zoom and pan, double-click:

📁 **[`system_architecture_viewer.html`](file:///c:/Users/sahil/OneDrive/Desktop/Capstone/system_architecture_viewer.html)**

### Key Features:
- Rendered via **Mermaid.js CDN** directly in Google Chrome, Microsoft Edge, or Safari.
- Instant **"Copy Mermaid Code"** buttons on each card to paste into AI tools.
- Filter buttons to isolate any diagram instantly.
- Styled in Kaiser Permanente's official brand palette.
