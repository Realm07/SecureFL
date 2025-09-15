document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let state = {
        selectedTaskId: null,
        tasks: {},
        network: {},
        tokenomics: {},
        charts: { accuracyChart: null },
        globe: null,
        _prevState: {} 
    };

    // --- DOM ELEMENT REFERENCES ---
    const eventLog = document.getElementById('event-log');
    const connectionStatusDot = document.getElementById('connection-status-dot');
    const connectionStatusText = document.getElementById('connection-status-text');
    const taskSelectContainer = document.getElementById('task-select-container');
    const selectedTaskDisplay = document.getElementById('selected-task-display');
    const taskOptionsList = document.getElementById('task-options-list');

    // --- LOGGING ---
    function logEvent(message, type = 'info') {
        const logEntry = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> <span class="log-message">${message}</span>`;
        logEntry.className = `log-entry log-${type}`;
        eventLog.prepend(logEntry);
    }

    // --- CHART INITIALIZATION ---
    function initializeAccuracyChart() {
        const ctx = document.getElementById('accuracy-chart').getContext('2d');
        state.charts.accuracyChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Metric', data: [], borderColor: '#8c7ae6', backgroundColor: 'rgba(140, 122, 230, 0.2)', fill: true, tension: 0.3 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: false }, x: { title: { display: true, text: 'Round' } } } }
        });
    }

    // --- SMART RENDER FUNCTION ---
    function render() {
        const prev = state._prevState;
        const curr = state;

        if (JSON.stringify(prev.tasks) !== JSON.stringify(curr.tasks)) {
            updateTaskSelector();
            updateTaskDetails();
            updateAccuracyChart();
            updateLiveAccuracy(); // NEW
            updateGlobeArcs(); 
            generateLiveLogsAndPulses(prev, curr); 
        }

        if (JSON.stringify(prev.network) !== JSON.stringify(curr.network)) {
            updateGlobePointsAndArcs();
        }

        if (JSON.stringify(prev.tokenomics) !== JSON.stringify(curr.tokenomics)) {
            updateNetworkEconomics();
        }
    }
    
    // --- GLOBE UPDATE LOGIC ---
    function updateGlobePointsAndArcs() {
        if (!state.globe) return;
        const connectedClients = state.network.connected_clients || [];
        state.globe.updateClientPoints(connectedClients);
        updateGlobeArcs();
    }

    function updateGlobeArcs() {
        if (!state.globe) return;
        const task = state.tasks[state.selectedTaskId];
        const serverLocation = task ? task.server_location : null;
        const connectedClients = state.network.connected_clients || [];
        
        state.globe.updateServerPoint(serverLocation);
        const connectedClientIds = connectedClients.map(c => c.id);

        connectedClients.forEach(client => {
            if (client.location && serverLocation) {
                state.globe.addOrUpdateArc(client.id, client.location, serverLocation);
            }
        });
        state.globe.removeInactiveArcs(connectedClientIds);
    }

    // --- UI UPDATE FUNCTIONS ---
    function updateTaskSelector() {
        const currentTaskIds = Object.keys(state.tasks);
        if (currentTaskIds.length === 0) return;

        // Set initial task if not set
        if (!state.selectedTaskId && currentTaskIds.length > 0) {
            state.selectedTaskId = currentTaskIds[0];
        }

        // Update display
        const selectedTaskName = state.selectedTaskId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        selectedTaskDisplay.querySelector('span').textContent = selectedTaskName;

        // Rebuild options list
        taskOptionsList.innerHTML = '';
        currentTaskIds.forEach(taskId => {
            const option = document.createElement('div');
            option.className = 'task-option';
            if (taskId === state.selectedTaskId) {
                option.classList.add('selected');
            }
            option.dataset.value = taskId;
            option.textContent = taskId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            taskOptionsList.appendChild(option);
        });
    }

    function updateTaskDetails() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) return;
        const task = state.tasks[state.selectedTaskId];
        document.getElementById('task-model-name').textContent = task.model_name || '--';
        document.getElementById('task-privacy-profile').textContent = (task.privacy_profile || '--').toUpperCase().replace('_', ' + ');
        document.getElementById('task-status').textContent = task.status || '--';
        document.getElementById('task-progress').textContent = `${task.current_round || 0} / ${task.total_rounds || 0} Rounds`;
    }
    
    // NEW: Function to update the Live Accuracy card
    function updateLiveAccuracy() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) return;
        const task = state.tasks[state.selectedTaskId];
        const history = task.metric_history;
        const latestAccuracyEl = document.getElementById('latest-accuracy');
        const latestAccuracyLabelEl = document.getElementById('latest-accuracy-label');
        
        latestAccuracyLabelEl.textContent = `Latest Round ${task.metric.toUpperCase()}`;
        
        if (history && history.length > 0) {
            const latestValue = history[history.length - 1];
            latestAccuracyEl.textContent = latestValue.toFixed(2);
        } else {
            latestAccuracyEl.textContent = '--';
        }
    }

    function updateAccuracyChart() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId] || !state.charts.accuracyChart) return;
        const task = state.tasks[state.selectedTaskId];
        const chart = state.charts.accuracyChart;
        const metricLabel = task.metric.toUpperCase();
        document.getElementById('accuracy-chart-header').textContent = `Global Model Performance`;
        const history = task.metric_history;
        chart.data.labels = history.map((_, i) => i);
        chart.data.datasets[0].label = metricLabel;
        chart.data.datasets[0].data = history;
        chart.options.scales.y.beginAtZero = (metricLabel !== 'RMSE');
        chart.update();
    }

    function updateNetworkEconomics() {
        document.getElementById('connected-clients-count').textContent = state.network.connected_clients?.length || 0;
        const leaderboardBody = document.querySelector("#leaderboard-table tbody");
        leaderboardBody.innerHTML = '';
        let totalStake = 0;
        if (state.tokenomics) {
            Object.entries(state.tokenomics).forEach(([clientId, account]) => {
                const row = leaderboardBody.insertRow();
                row.innerHTML = `<td>${clientId}</td><td>${account.stake.toFixed(2)}</td><td>${account.balance.toFixed(2)}</td>`;
                totalStake += account.stake;
            });
        }
        document.getElementById('total-stake').textContent = `${totalStake.toFixed(2)} PHOENIX`;
    }

    function generateLiveLogsAndPulses(prevState, currentState) {
        if (!prevState.tasks || Object.keys(prevState.tasks).length === 0) return;

        for (const taskId in currentState.tasks) {
            const prevTask = prevState.tasks[taskId] || { current_round: 0, status: '', metric_history: [], selected_clients: [] };
            const currentTask = currentState.tasks[taskId];
            
            if (currentTask.current_round > prevTask.current_round) {
                const metricName = currentTask.metric.toUpperCase();
                const latestMetric = currentTask.metric_history[currentTask.metric_history.length - 1];
                logEvent(`Task '${taskId}' round ${prevTask.current_round} complete. ${metricName}: ${latestMetric.toFixed(2)}`);
            }
            if (currentTask.status !== prevTask.status) {
                 logEvent(`Task '${taskId}' status changed to: ${currentTask.status}`);
            }

            if (state.selectedTaskId === taskId && state.globe) {
                const prevReadyClients = prevTask.selected_clients || [];
                const currentReadyClients = currentTask.selected_clients || [];
                const newlyReadyClients = currentReadyClients.filter(id => !prevReadyClients.includes(id));
                
                newlyReadyClients.forEach(clientId => {
                    logEvent(`Client #${clientId} finished training for task '${taskId}'. Sending update.`, 'success');
                    state.globe.triggerPulse(clientId);
                });
            }
        }

        const prevClients = (prevState.network.connected_clients || []).map(c => c.id);
        const currentClients = (currentState.network.connected_clients || []).map(c => c.id);
        const connected = currentClients.filter(id => !prevClients.includes(id));
        const disconnected = prevClients.filter(id => !currentClients.includes(id));
        connected.forEach(id => logEvent(`Client #${id} connected.`));
        disconnected.forEach(id => logEvent(`Client #${id} disconnected.`, 'warn'));
    }

    async function fetchData() {
        try {
            const [statusResponse, tokenomicsResponse] = await Promise.all([
                fetch('/status'),
                fetch('/tokenomics')
            ]);

            if (!statusResponse.ok || !tokenomicsResponse.ok) throw new Error('Network response was not ok');
            
            const statusData = await statusResponse.json();
            const tokenomicsData = await tokenomicsResponse.json();
            
            state._prevState = JSON.parse(JSON.stringify({ tasks: state.tasks, network: state.network, tokenomics: state.tokenomics }));
            state.tasks = statusData.tasks;
            state.network = statusData.network_info;
            state.tokenomics = tokenomicsData;
            
            render(); 
            
            connectionStatusDot.className = 'status-dot connected';
            connectionStatusText.textContent = 'Connected';
        } catch (error) {
            connectionStatusDot.className = 'status-dot disconnected';
            connectionStatusText.textContent = 'Disconnected';
            console.error("Fetch error:", error);
        }
    }
    
    // --- EVENT LISTENERS for Custom Select ---
    selectedTaskDisplay.addEventListener('click', () => {
        taskSelectContainer.classList.toggle('open');
    });

    taskOptionsList.addEventListener('click', (e) => {
        const option = e.target.closest('.task-option');
        if (!option) return;

        const newTaskId = option.dataset.value;
        if (newTaskId !== state.selectedTaskId) {
            state.selectedTaskId = newTaskId;
            if (state.globe) {
                state.globe.clearAllArcs();
                updateGlobeArcs();
            }
            updateTaskSelector();
            updateTaskDetails();
            updateAccuracyChart();
            updateLiveAccuracy();
        }
        taskSelectContainer.classList.remove('open');
    });
    
    // Close dropdown if clicked outside
    window.addEventListener('click', (e) => {
        if (!taskSelectContainer.contains(e.target)) {
            taskSelectContainer.classList.remove('open');
        }
    });

    // --- INITIALIZATION ---
    function initializeDashboard() {
        logEvent('Dashboard Initialized. Connecting to server...');
        if (typeof Chart === 'undefined' || typeof THREE === 'undefined' || typeof createGlobe === 'undefined') {
            logEvent('Error: A required library failed to load.', 'error');
            return;
        }

        initializeAccuracyChart();
        const globeContainer = document.getElementById('globe-container');
        if (globeContainer) {
            state.globe = createGlobe(globeContainer);
        }
        
        if (document.visibilityState === 'visible') {
            startPolling();
        }
    }

    let fetchDataInterval;
    function startPolling() {
        if (fetchDataInterval) clearInterval(fetchDataInterval);
        fetchData();
        fetchDataInterval = setInterval(fetchData, 5000);
    }
    function stopPolling() {
        clearInterval(fetchDataInterval);
    }
    document.addEventListener('visibilitychange', () => {
        document.visibilityState === 'visible' ? startPolling() : stopPolling();
    });
    
    initializeDashboard();
});