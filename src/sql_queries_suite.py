"""
Kaiser Permanente Star Schema Analytical SQL Queries Suite
Encapsulates 11 production queries demonstrating all 9 ANSI-SQL techniques required by Project 3:
- SELECT, WHERE, CASE Statements, GROUP BY
- INNER JOIN, LEFT JOIN, Subqueries
- Window Functions (SUM, AVG OVER), Ranking Functions (DENSE_RANK OVER)
"""

SQL_QUERIES = [
    {
        "id": "q1",
        "title": "Query 1: Executive Contact Center KPI Overview",
        "purpose": "Calculates baseline operational metrics: total interaction volume, unique active members, agent headcount, overall escalation rate, First Contact Resolution (FCR), mean resolution time, and mean CSAT.",
        "techniques": ["SELECT", "COUNT(DISTINCT)", "AVG", "CASE Statement", "Scalar Math"],
        "sql": """SELECT 
    COUNT(f.Interaction_ID) AS Total_Interactions,
    COUNT(DISTINCT f.Member_ID) AS Unique_Members,
    COUNT(DISTINCT f.Agent_ID) AS Active_Agents,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Rate_Pct,
    ROUND(AVG(f.Resolution_Time), 2) AS Mean_Resolution_Time_Min,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Mean_CSAT_Score,
    ROUND(SUM(CASE WHEN f.Escalation_Flag = 1 THEN 1 ELSE 0 END), 0) AS Total_Escalated_Calls
FROM Fact_Interactions_Enriched f;"""
    },
    {
        "id": "q2",
        "title": "Query 2: Contact Channel Disparity & Escalation Analysis",
        "purpose": "Bivariate analysis confirming Hypothesis 3: Voice vs Digital Channel disparity. Evaluates volume share, escalation rate, FCR, and average duration across all contact channels.",
        "techniques": ["SELECT", "GROUP BY", "ORDER BY", "Subquery in SELECT", "Aggregation"],
        "sql": """SELECT 
    f.Contact_Channel,
    COUNT(f.Interaction_ID) AS Channel_Volume,
    ROUND(COUNT(f.Interaction_ID) * 100.0 / (SELECT COUNT(*) FROM Fact_Interactions_Enriched), 2) AS Volume_Share_Pct,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Rate_Pct,
    ROUND(AVG(f.Resolution_Time), 2) AS Avg_Duration_Min,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT
FROM Fact_Interactions_Enriched f
GROUP BY f.Contact_Channel
ORDER BY Escalation_Rate_Pct DESC;"""
    },
    {
        "id": "q3",
        "title": "Query 3: Service Friction Index (SFI) Quintile Impact",
        "purpose": "Segments interactions into 5 Service Friction Index quintiles to demonstrate that operational friction exponentially drives member escalations and suppresses CSAT.",
        "techniques": ["SELECT", "CASE Statements (Categorization)", "GROUP BY", "ORDER BY"],
        "sql": """SELECT 
    CASE 
        WHEN f.Service_Friction_Index < 0.20 THEN '1. Minimal Friction (0.00 - 0.19)'
        WHEN f.Service_Friction_Index < 0.40 THEN '2. Low Friction (0.20 - 0.39)'
        WHEN f.Service_Friction_Index < 0.60 THEN '3. Moderate Friction (0.40 - 0.59)'
        WHEN f.Service_Friction_Index < 0.80 THEN '4. High Friction (0.60 - 0.79)'
        ELSE '5. Critical Friction (0.80 - 1.00)'
    END AS Friction_Tier,
    COUNT(f.Interaction_ID) AS Total_Volume,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Rate_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT,
    ROUND(AVG(f.Resolution_Time), 2) AS Avg_Resolution_Time_Min
FROM Fact_Interactions_Enriched f
GROUP BY Friction_Tier
ORDER BY Friction_Tier ASC;"""
    },
    {
        "id": "q4",
        "title": "Query 4: Member Behavioral Persona Penetration & Risk",
        "purpose": "Joins Dim_Member_Enriched with Fact table to profile the 4 K-Means personas on demographic age, digital adoption propensity, interaction volume, and escalation rates.",
        "techniques": ["SELECT", "LEFT JOIN", "GROUP BY", "ORDER BY", "Multi-table Aggregation"],
        "sql": """SELECT 
    m.Member_Segment AS Persona_Name,
    COUNT(DISTINCT m.Member_ID) AS Total_Members,
    ROUND(AVG(m.Age), 1) AS Avg_Member_Age,
    ROUND(AVG(m.Digital_Adoption_Score) * 100.0, 1) AS Avg_Digital_Adoption_Pct,
    COUNT(f.Interaction_ID) AS Total_Interactions,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Interaction_Escalation_Rate_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Segment_Avg_CSAT,
    ROUND(AVG(f.Service_Friction_Index), 3) AS Segment_Avg_Friction
FROM Dim_Member_Enriched m
LEFT JOIN Fact_Interactions_Enriched f ON m.Member_ID = f.Member_ID
GROUP BY m.Member_Segment
ORDER BY Interaction_Escalation_Rate_Pct DESC;"""
    },
    {
        "id": "q5",
        "title": "Query 5: Regional Performance & Plan Type Breakdown",
        "purpose": "Cross-tabulates Kaiser Permanente geographic regions (Northern CA, Southern CA, Colorado, etc.) and commercial/Medicare plans to identify regional escalation hotspots.",
        "techniques": ["SELECT", "INNER JOIN", "Multi-column GROUP BY", "ORDER BY"],
        "sql": """SELECT 
    m.Region,
    m.Plan_Type,
    COUNT(DISTINCT m.Member_ID) AS Member_Count,
    COUNT(f.Interaction_ID) AS Interaction_Volume,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Pct
FROM Dim_Member_Enriched m
JOIN Fact_Interactions_Enriched f ON m.Member_ID = f.Member_ID
GROUP BY m.Region, m.Plan_Type
ORDER BY m.Region ASC, Escalation_Rate_Pct DESC;"""
    },
    {
        "id": "q6",
        "title": "Query 6: Department & Queue Operational SLA Heatmap",
        "purpose": "Joins Dim_Agent with Fact table to pinpoint service queues experiencing the longest handle times, highest friction, and highest model-predicted escalation probabilities.",
        "techniques": ["SELECT", "INNER JOIN", "GROUP BY", "WHERE / HAVING", "ORDER BY"],
        "sql": """SELECT 
    a.Department,
    a.Queue,
    COUNT(f.Interaction_ID) AS Queue_Volume,
    ROUND(AVG(f.Resolution_Time), 2) AS Mean_Resolution_Time_Min,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Rate_Pct,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.Predicted_Escalation_Prob) * 100.0, 2) AS Avg_Model_Escalation_Prob_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT
FROM Dim_Agent a
JOIN Fact_Interactions_Enriched f ON a.Agent_ID = f.Agent_ID
GROUP BY a.Department, a.Queue
ORDER BY Escalation_Rate_Pct DESC;"""
    },
    {
        "id": "q7",
        "title": "Query 7: High-Risk Member Escalation Early-Warning Roster",
        "purpose": "Generates a real-time operational roster of members flagged by our predictive ML model with high escalation probability (prob >= 0.70) for proactive outbound outreach.",
        "techniques": ["SELECT", "INNER JOIN", "WHERE (Compound Filters)", "ORDER BY", "LIMIT"],
        "sql": """SELECT 
    f.Interaction_ID,
    f.Member_ID,
    m.Age AS Member_Age,
    m.Plan_Type,
    m.Region,
    m.Member_Segment,
    f.Contact_Channel,
    f.Interaction_Reason,
    ROUND(f.Service_Friction_Index, 3) AS Friction_Index,
    ROUND(f.Predicted_Escalation_Prob, 3) AS Escalation_Prob,
    f.Dominant_Topic,
    ROUND(f.Sentiment_Score, 3) AS Sentiment_Score
FROM Fact_Interactions_Enriched f
JOIN Dim_Member_Enriched m ON f.Member_ID = m.Member_ID
WHERE f.Predicted_Escalation_Prob >= 0.70
  AND f.Escalation_Flag = 1
ORDER BY f.Predicted_Escalation_Prob DESC
LIMIT 50;"""
    },
    {
        "id": "q8",
        "title": "Query 8: NLP Dominant Topics & Sentiment Distribution",
        "purpose": "Evaluates interaction volumes, deep learning sentiment scores, and escalation rates across the 5 Kaiser operational themes discovered by Latent Dirichlet Allocation (LDA).",
        "techniques": ["SELECT", "GROUP BY", "AVG Aggregation", "ORDER BY"],
        "sql": """SELECT 
    f.Dominant_Topic,
    COUNT(f.Interaction_ID) AS Topic_Volume,
    ROUND(AVG(f.Sentiment_Score), 3) AS Avg_Deep_Learning_Sentiment,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT,
    ROUND(AVG(f.Resolution_Time), 2) AS Avg_Duration_Min
FROM Fact_Interactions_Enriched f
GROUP BY f.Dominant_Topic
ORDER BY Topic_Volume DESC;"""
    },
    {
        "id": "q9",
        "title": "Query 9: Agent Scorecard with Window Functions & Ranking",
        "purpose": "Demonstrates advanced ANSI-SQL Window and Ranking Functions (DENSE_RANK OVER and AVG OVER) to benchmark contact center agents against department-wide CSAT averages.",
        "techniques": ["Window Functions: DENSE_RANK() OVER", "Window Functions: AVG() OVER", "PARTITION BY", "HAVING", "INNER JOIN"],
        "sql": """SELECT 
    a.Agent_ID,
    a.Department,
    a.Queue,
    COUNT(f.Interaction_ID) AS Handled_Calls,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS Agent_Actual_FCR_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Agent_Actual_CSAT,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Agent_Escalation_Rate_Pct,
    DENSE_RANK() OVER (PARTITION BY a.Department ORDER BY AVG(f.Escalation_Flag) DESC) AS Dept_Escalation_Rank,
    ROUND(AVG(AVG(f.Satisfaction_Score)) OVER (PARTITION BY a.Department), 2) AS Dept_Benchmark_CSAT
FROM Dim_Agent a
JOIN Fact_Interactions_Enriched f ON a.Agent_ID = f.Agent_ID
GROUP BY a.Agent_ID, a.Department, a.Queue
HAVING COUNT(f.Interaction_ID) >= 20
ORDER BY a.Department, Dept_Escalation_Rank ASC
LIMIT 30;"""
    },
    {
        "id": "q10",
        "title": "Query 10: RAG 30% Call Deflection ROI Financial Model",
        "purpose": "Calculates the annual cost avoidance of deflecting 30% of routine Pharmacy, Claims, and Booking voice calls to autonomous MediRAG navigation ($7.50 voice cost vs $0.65 digital).",
        "techniques": ["Common Table Expressions (WITH CTE)", "Mathematical Projections", "Filtering", "Subquery Logic"],
        "sql": """WITH RoutineInquiries AS (
    SELECT 
        COUNT(Interaction_ID) AS Total_Routine_Volume,
        COUNT(CASE WHEN Contact_Channel = 'Phone' THEN 1 END) AS Voice_Routine_Volume
    FROM Fact_Interactions_Enriched
    WHERE Interaction_Reason IN (
        'Prescription Refill',
        'Appointment Scheduling',
        'Billing Question',
        'Claims Status',
        'General Inquiry'
    )
)
SELECT 
    Total_Routine_Volume,
    Voice_Routine_Volume,
    ROUND(Voice_Routine_Volume * 0.30, 0) AS Deflected_Calls_30Pct,
    ROUND(Voice_Routine_Volume * 0.30 * 7.50, 2) AS Gross_Voice_Cost_Avoided_USD,
    ROUND(Voice_Routine_Volume * 0.30 * 0.65, 2) AS RAG_Inference_Infrastructure_Cost_USD,
    ROUND((Voice_Routine_Volume * 0.30 * 7.50) - (Voice_Routine_Volume * 0.30 * 0.65), 2) AS Net_Annual_Cost_Savings_USD
FROM RoutineInquiries;"""
    },
    {
        "id": "q11",
        "title": "Query 11: Star Schema Referential Integrity Audit",
        "purpose": "Executes UNION ALL orphan record verification across all dimension tables to validate 100% referential integrity with Fact_Interactions_Enriched.",
        "techniques": ["UNION ALL", "LEFT JOIN", "IS NULL checks", "Schema Audit"],
        "sql": """SELECT 
    'Orphan Member Records' AS Integrity_Check,
    COUNT(f.Interaction_ID) AS Orphan_Count
FROM Fact_Interactions_Enriched f
LEFT JOIN Dim_Member_Enriched m ON f.Member_ID = m.Member_ID
WHERE m.Member_ID IS NULL

UNION ALL

SELECT 
    'Orphan Agent Records' AS Integrity_Check,
    COUNT(f.Interaction_ID) AS Orphan_Count
FROM Fact_Interactions_Enriched f
LEFT JOIN Dim_Agent a ON f.Agent_ID = a.Agent_ID
WHERE a.Agent_ID IS NULL

UNION ALL

SELECT 
    'Orphan Date Records' AS Integrity_Check,
    COUNT(f.Interaction_ID) AS Orphan_Count
FROM Fact_Interactions_Enriched f
LEFT JOIN Dim_Date d ON f.Date_Key = d.Date_Key
WHERE d.Date_Key IS NULL;"""
    }
]
