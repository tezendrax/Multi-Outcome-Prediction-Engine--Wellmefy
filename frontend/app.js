document.addEventListener("DOMContentLoaded", () => {
    // API Server Configuration
    const BASE_URL = window.location.origin;

    // DOM Elements
    const serverStatus = document.getElementById("server-status");
    const studentSelect = document.getElementById("student-id-select");
    const customIdGroup = document.getElementById("custom-id-group");
    const customIdInput = document.getElementById("student-id-input");
    
    const weekSlider = document.getElementById("semester-week");
    const weekDisplay = document.getElementById("week-display");
    
    const btnFetch = document.getElementById("btn-fetch");
    const btnTrigger = document.getElementById("btn-trigger");
    
    // Model versions elements
    const infoBurnoutV = document.getElementById("info-burnout-v");
    const infoAnxietyV = document.getElementById("info-anxiety-v");
    const infoDepressiveV = document.getElementById("info-depressive-v");
    
    // Result elements
    const statusBanner = document.getElementById("status-banner");
    
    const burnoutGauge = document.getElementById("burnout-gauge");
    const burnoutValue = document.getElementById("burnout-value");
    const burnoutPill = document.getElementById("burnout-pill");
    
    const anxietyBadge = document.getElementById("anxiety-badge");
    const anxietyScoreText = document.getElementById("anxiety-score");
    const anxietyProgress = document.getElementById("anxiety-progress");
    
    const depressiveValueText = document.getElementById("depressive-value");
    const depressiveProgress = document.getElementById("depressive-progress");
    
    // Diagnostic detail elements
    const diagThreshold = document.getElementById("diag-threshold");
    const diagAnomaly = document.getElementById("diag-anomaly");
    const diagVersion = document.getElementById("diag-version");
    const diagTime = document.getElementById("diag-time");

    // ==========================================================================
    // Event Handlers
    // ==========================================================================

    // Show/hide custom student ID input
    studentSelect.addEventListener("change", (e) => {
        if (e.target.value === "custom") {
            customIdGroup.classList.remove("hidden");
        } else {
            customIdGroup.classList.add("hidden");
        }
    });

    // Update semester week display
    weekSlider.addEventListener("input", (e) => {
        weekDisplay.textContent = `Week ${e.target.value}`;
    });

    // Run Projections button click
    btnFetch.addEventListener("click", () => {
        const studentId = getActiveStudentId();
        if (!studentId) return alert("Please enter a valid Student ID.");
        
        fetchProjections(studentId, parseInt(weekSlider.value));
    });

    // Force Recalculate button click
    btnTrigger.addEventListener("click", () => {
        const studentId = getActiveStudentId();
        if (!studentId) return alert("Please enter a valid Student ID.");
        
        triggerProjections(studentId, parseInt(weekSlider.value));
    });

    // Helper: get selected student id
    function getActiveStudentId() {
        if (studentSelect.value === "custom") {
            return customIdInput.value.trim();
        }
        return studentSelect.value;
    }

    // ==========================================================================
    // API Services
    // ==========================================================================

    // Fetch API Health & Active Models on load
    async function checkServerHealth() {
        try {
            const res = await fetch(`${BASE_URL}/health`);
            if (!res.ok) throw new Error("Server unhealthy");
            
            const data = await res.json();
            
            // Update Status Badge to online
            serverStatus.innerHTML = `
                <span class="status-indicator online"></span>
                <span class="status-text">MOPE Connected</span>
            `;
            
            // Set model details
            const registered = data.registered_models || {};
            infoBurnoutV.textContent = registered.burnout_xgb ? registered.burnout_xgb.version : "v1.4.2";
            infoAnxietyV.textContent = registered.anxiety_xgb ? registered.anxiety_xgb.version : "v1.4.2";
            infoDepressiveV.textContent = registered.depressive_lgb ? registered.depressive_lgb.version : "v1.4.2";
            
            if (data.active_version) {
                diagVersion.textContent = data.active_version;
            }
        } catch (err) {
            console.error("Health check error:", err);
            serverStatus.innerHTML = `
                <span class="status-indicator offline"></span>
                <span class="status-text">MOPE Offline</span>
            `;
            // Set mock active versions as fallback representation
            infoBurnoutV.textContent = "v1.4.2 (Offline)";
            infoAnxietyV.textContent = "v1.4.2 (Offline)";
            infoDepressiveV.textContent = "v1.4.2 (Offline)";
        }
    }

    // Fetch cached predictions (GET)
    async function fetchProjections(studentId, week) {
        setLoading(true);
        try {
            const res = await fetch(`${BASE_URL}/api/v1/predictions/mope?student_id=${studentId}&semester_week=${week}`);
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Prediction request failed");
            }
            const data = await res.json();
            updateDashboard(data);
        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    // Force recalculate predictions (POST)
    async function triggerProjections(studentId, week) {
        setLoading(true);
        try {
            const payload = {
                student_id: studentId,
                temporal_metadata: {
                    semester_week: week
                }
            };
            
            const res = await fetch(`${BASE_URL}/api/v1/predictions/mope/trigger`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Trigger request failed");
            }
            const data = await res.json();
            updateDashboard(data);
        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    // ==========================================================================
    // UI Rendering
    // ==========================================================================

    function setLoading(isLoading) {
        if (isLoading) {
            btnFetch.disabled = true;
            btnTrigger.disabled = true;
            btnFetch.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;
        } else {
            btnFetch.disabled = false;
            btnTrigger.disabled = false;
            btnFetch.innerHTML = `<i class="fa-solid fa-chart-line"></i> Run Projections`;
        }
    }

    function updateDashboard(data) {
        // Extract data fields
        const burnoutRisk = data.predictions.burnout_risk;
        const anxietyScore = data.predictions.anxiety_score;
        const alertTriggered = data.predictions.clinical_indicator_alert;
        
        const details = data.details || {};
        const burnoutProb = details.burnout_probability || burnoutRisk;
        const anxietyLevel = details.anxiety_level_risk || "Low";
        const depressiveIndex = details.depressive_onset_index || 0.0;
        const thresholdBreached = details.critical_threshold_breached || false;
        const anomalyWarning = details.anomaly_warning || false;
        
        const version = data.model_version;
        const timestamp = new Date(data.timestamp).toLocaleString();

        // 1. Update Status Banner
        statusBanner.className = "status-banner";
        if (alertTriggered) {
            statusBanner.classList.add("alert-danger");
            statusBanner.querySelector(".banner-icon").innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i>`;
            statusBanner.querySelector("h3").textContent = "Status: CRITICAL SAFETY ALERT";
            statusBanner.querySelector("p").textContent = "Student demonstrates signs of severe burnout or emotional onset. Interventions recommended.";
        } else {
            statusBanner.classList.add("alert-safe");
            statusBanner.querySelector(".banner-icon").innerHTML = `<i class="fa-solid fa-circle-check"></i>`;
            statusBanner.querySelector("h3").textContent = "Status: STABLE / HEALTHY";
            statusBanner.querySelector("p").textContent = "Student wellness indicators currently remain within safe baseline bounds.";
        }

        // 2. Update Burnout Radial Gauge
        const burnoutPercent = Math.round(burnoutProb * 100);
        burnoutGauge.style.setProperty("--value", burnoutPercent);
        burnoutValue.textContent = `${burnoutPercent}%`;
        
        // Gauge colors and pill status
        let burnoutColor = "#10b981"; // green
        let burnoutStatus = "Healthy";
        if (burnoutProb >= 0.70) {
            burnoutColor = "#ef4444"; // red
            burnoutStatus = "Critical";
            burnoutPill.className = "risk-pill text-danger";
        } else if (burnoutProb >= 0.40) {
            burnoutColor = "#f59e0b"; // yellow
            burnoutStatus = "Elevated";
            burnoutPill.className = "risk-pill text-warning";
        } else {
            burnoutPill.className = "risk-pill text-success";
        }
        burnoutGauge.style.setProperty("--gauge-color", burnoutColor);
        burnoutPill.textContent = burnoutStatus;

        // 3. Update Anxiety Card
        anxietyBadge.textContent = anxietyLevel;
        anxietyBadge.className = "anxiety-level-badge";
        
        let anxietyColor = "#10b981";
        if (anxietyLevel === "High") {
            anxietyColor = "#ef4444";
            anxietyBadge.classList.add("text-danger");
        } else if (anxietyLevel === "Medium") {
            anxietyColor = "#f59e0b";
            anxietyBadge.classList.add("text-warning");
        } else {
            anxietyBadge.classList.add("text-success");
        }
        
        anxietyScoreText.textContent = anxietyScore.toFixed(2);
        anxietyProgress.style.width = `${Math.round(anxietyScore * 100)}%`;
        anxietyProgress.style.setProperty("--progress-color", anxietyColor);
        anxietyProgress.style.setProperty("--progress-color-glow", `${anxietyColor}40`);

        // 4. Update Depressive Onset Card
        depressiveValueText.textContent = depressiveIndex.toFixed(2);
        depressiveProgress.style.width = `${Math.round(depressiveIndex * 100)}%`;
        
        let depressiveColor = "#06b6d4"; // cyan
        if (depressiveIndex >= 0.60) {
            depressiveColor = "#ef4444"; // red
            depressiveValueText.style.background = "linear-gradient(135deg, #ef4444, #f59e0b)";
            depressiveValueText.style.webkitBackgroundClip = "text";
        } else {
            depressiveValueText.style.background = "linear-gradient(135deg, #00ADB5, #FF2E93)";
            depressiveValueText.style.webkitBackgroundClip = "text";
        }
        depressiveProgress.style.setProperty("--progress-color", depressiveColor);
        depressiveProgress.style.setProperty("--progress-color-glow", `${depressiveColor}40`);

        // 5. Update Bottom Diagnostics reports
        if (thresholdBreached) {
            diagThreshold.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-danger"></i> <span class="text-danger">Yes, Breached</span>`;
        } else {
            diagThreshold.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> <span class="text-success">No, Safe</span>`;
        }

        if (anomalyWarning) {
            diagAnomaly.innerHTML = `<i class="fa-solid fa-bolt-lightning text-warning"></i> <span class="text-warning">Yes, Spike Detected</span>`;
        } else {
            diagAnomaly.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> <span class="text-success">No Anomalies</span>`;
        }

        diagVersion.textContent = version;
        diagTime.textContent = timestamp;
    }

    // Initialize health check on loading
    checkServerHealth();
});
