-- =============================================================================
-- KAISER PERMANENTE MEMBER EXPERIENCE INTELLIGENCE PLATFORM
-- Production Star Schema Analytical SQL Queries & Reporting Suite
--
-- Target Databases: Neon Cloud PostgreSQL / AWS RDS / SQLite (ANSI SQL Compliant)
-- Star Schema Tables:
--   - Fact_Interactions_Enriched (Fact table, 10,000 interactions)
--   - Dim_Member_Enriched (Dimension, 2,621 unique members)
--   - Dim_Agent (Dimension, 250 contact center agents)
--   - Dim_Date (Dimension, calendar dates YYYYMMDD)
--   - RAG_Corpus_Staging (Sanitized clinical transcripts & guidelines)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- QUERY 1: EXECUTIVE CONTACT CENTER KPI OVERVIEW
-- Demonstrates: SELECT, COUNT(DISTINCT), AVG, CASE Statements
-- -----------------------------------------------------------------------------
SELECT 
    COUNT(f.Interaction_ID) AS Total_Interactions,
    COUNT(DISTINCT f.Member_ID) AS Unique_Members,
    COUNT(DISTINCT f.Agent_ID) AS Active_Agents,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Rate_Pct,
    ROUND(AVG(f.Resolution_Time), 2) AS Mean_Resolution_Time_Min,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Mean_CSAT_Score,
    ROUND(SUM(CASE WHEN f.Escalation_Flag = 1 THEN 1 ELSE 0 END), 0) AS Total_Escalated_Calls
FROM Fact_Interactions_Enriched f;


-- -----------------------------------------------------------------------------
-- QUERY 2: CONTACT CHANNEL DISPARITY & ESCALATION ANALYSIS
-- Demonstrates: GROUP BY, ORDER BY, Subquery in SELECT, Aggregation
-- -----------------------------------------------------------------------------
SELECT 
    f.Contact_Channel,
    COUNT(f.Interaction_ID) AS Channel_Volume,
    ROUND(COUNT(f.Interaction_ID) * 100.0 / (SELECT COUNT(*) FROM Fact_Interactions_Enriched), 2) AS Volume_Share_Pct,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.First_Contact_Resolution) * 100.0, 2) AS FCR_Rate_Pct,
    ROUND(AVG(f.Resolution_Time), 2) AS Avg_Duration_Min,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT
FROM Fact_Interactions_Enriched f
GROUP BY f.Contact_Channel
ORDER BY Escalation_Rate_Pct DESC;


-- -----------------------------------------------------------------------------
-- QUERY 3: SERVICE FRICTION INDEX (SFI) QUINTILE IMPACT
-- Demonstrates: Multi-branch CASE Statements, GROUP BY, ORDER BY
-- -----------------------------------------------------------------------------
SELECT 
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
ORDER BY Friction_Tier ASC;


-- -----------------------------------------------------------------------------
-- QUERY 4: MEMBER BEHAVIORAL PERSONA PENETRATION & RISK
-- Demonstrates: LEFT JOIN, GROUP BY, Multi-table Aggregation
-- -----------------------------------------------------------------------------
SELECT 
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
ORDER BY Interaction_Escalation_Rate_Pct DESC;


-- -----------------------------------------------------------------------------
-- QUERY 5: REGIONAL PERFORMANCE & HEALTH PLAN BREAKDOWN
-- Demonstrates: INNER JOIN, Multi-column GROUP BY, ORDER BY
-- -----------------------------------------------------------------------------
SELECT 
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
ORDER BY m.Region ASC, Escalation_Rate_Pct DESC;


-- -----------------------------------------------------------------------------
-- QUERY 6: DEPARTMENT & QUEUE OPERATIONAL SLA HEATMAP
-- Demonstrates: INNER JOIN, GROUP BY, SLA Escalation Heatmap
-- -----------------------------------------------------------------------------
SELECT 
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
ORDER BY Escalation_Rate_Pct DESC;


-- -----------------------------------------------------------------------------
-- QUERY 7: HIGH-RISK MEMBER ESCALATION EARLY-WARNING ROSTER
-- Demonstrates: INNER JOIN, Compound WHERE filters, ORDER BY, LIMIT
-- -----------------------------------------------------------------------------
SELECT 
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
LIMIT 50;


-- -----------------------------------------------------------------------------
-- QUERY 8: TOP NLP DOMINANT TOPICS & COMPLAINT REASONS
-- Demonstrates: GROUP BY, AVG, Topic Sentiment Distribution
-- -----------------------------------------------------------------------------
SELECT 
    f.Dominant_Topic,
    COUNT(f.Interaction_ID) AS Topic_Volume,
    ROUND(AVG(f.Sentiment_Score), 3) AS Avg_Deep_Learning_Sentiment,
    ROUND(AVG(f.Escalation_Flag) * 100.0, 2) AS Escalation_Rate_Pct,
    ROUND(AVG(f.Satisfaction_Score), 2) AS Avg_CSAT,
    ROUND(AVG(f.Resolution_Time), 2) AS Avg_Duration_Min
FROM Fact_Interactions_Enriched f
GROUP BY f.Dominant_Topic
ORDER BY Topic_Volume DESC;


-- -----------------------------------------------------------------------------
-- QUERY 9: AGENT PERFORMANCE SCORECARD WITH WINDOW & RANKING FUNCTIONS
-- Demonstrates: DENSE_RANK() OVER, AVG() OVER, PARTITION BY, HAVING, INNER JOIN
-- -----------------------------------------------------------------------------
SELECT 
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
LIMIT 30;


-- -----------------------------------------------------------------------------
-- QUERY 10: RAG CARE NAVIGATION CALL DEFLECTION ROI FINANCIAL MODEL
-- Demonstrates: Common Table Expression (WITH CTE), Financial Math Modeling
-- -----------------------------------------------------------------------------
WITH RoutineInquiries AS (
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
FROM RoutineInquiries;


-- -----------------------------------------------------------------------------
-- QUERY 11: STAR SCHEMA REFERENTIAL INTEGRITY AUDIT
-- Demonstrates: UNION ALL, LEFT JOIN, IS NULL Orphan Verification
-- -----------------------------------------------------------------------------
SELECT 
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
WHERE d.Date_Key IS NULL;
