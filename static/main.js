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
    const taskSelect = document.getElementById('task-select');
    const eventLog = document.getElementById('event-log');
    const connectionStatusDot = document.getElementById('connection-status-dot');
    const connectionStatusText = document.getElementById('connection-status-text');


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
            updateGlobeArcs(); // Arcs depend on task (server location)
            generateLiveLogsAndPulses(prev, curr); 
        }

        if (JSON.stringify(prev.network) !== JSON.stringify(curr.network)) {
            updateGlobePointsAndArcs(); // Points and arcs depend on connected clients
        }

        if (JSON.stringify(prev.tokenomics) !== JSON.stringify(curr.tokenomics)) {
            updateNetworkEconomics();
        }
    }
    
    // --- REFINED GLOBE UPDATE LOGIC ---
    function updateGlobePointsAndArcs() {
        if (!state.globe) return;
        const connectedClients = state.network.connected_clients || [];
        
        // 1. Update the client points (dots on cities)
        state.globe.updateClientPoints(connectedClients);
        
        // 2. Update arcs based on connection status
        updateGlobeArcs();
    }

    function updateGlobeArcs() {
        if (!state.globe) return;
        const task = state.tasks[state.selectedTaskId];
        const serverLocation = task ? task.server_location : null;
        const connectedClients = state.network.connected_clients || [];
        
        // Update the central server point
        state.globe.updateServerPoint(serverLocation);

        // Arcs are now based on who is CONNECTED, not who is SELECTED
        const connectedClientIds = connectedClients.map(c => c.id);

        // Add/update arcs for all currently connected clients to the current server
        connectedClients.forEach(client => {
            if (client.location && serverLocation) {
                state.globe.addOrUpdateArc(client.id, client.location, serverLocation);
            }
        });

        // Remove arcs for any client that has disconnected
        state.globe.removeInactiveArcs(connectedClientIds);
    }


    // --- UI UPDATE FUNCTIONS (No changes needed in these helpers) ---
    function updateTaskSelector() {
        const currentTaskIds = Object.keys(state.tasks);
        const existingOptionIds = Array.from(taskSelect.options).map(o => o.value);

        if (JSON.stringify(currentTaskIds) === JSON.stringify(existingOptionIds)) return;

        taskSelect.innerHTML = '';
        currentTaskIds.forEach(taskId => {
            const option = document.createElement('option');
            option.value = taskId;
            option.textContent = taskId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            taskSelect.appendChild(option);
        });
        if (!state.selectedTaskId && currentTaskIds.length > 0) {
            state.selectedTaskId = currentTaskIds[0];
        }
        taskSelect.value = state.selectedTaskId;
    }

    function updateTaskDetails() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) return;
        const task = state.tasks[state.selectedTaskId];
        document.getElementById('task-model-name').textContent = task.model_name || '--';
        document.getElementById('task-privacy-profile').textContent = (task.privacy_profile || '--').toUpperCase().replace('_', ' + ');
        document.getElementById('task-status').textContent = task.status || '--';
        document.getElementById('task-progress').textContent = `${task.current_round || 0} / ${task.total_rounds || 0} Rounds`;
    }

    function updateAccuracyChart() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId] || !state.charts.accuracyChart) return;
        const task = state.tasks[state.selectedTaskId];
        const chart = state.charts.accuracyChart;
        const metricLabel = task.metric.toUpperCase();
        document.getElementById('accuracy-chart-header').textContent = `Global Model ${metricLabel}`;
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

    // --- PULSE TRIGGER LOGIC ---
    function generateLiveLogsAndPulses(prevState, currentState) {
        if (!prevState.tasks || Object.keys(prevState.tasks).length === 0) return;

        for (const taskId in currentState.tasks) {
            const prevTask = prevState.tasks[taskId] || { current_round: 0, status: '', metric_history: [], selected_clients: [] };
            const currentTask = currentState.tasks[taskId];
            
            // Log round completion
            if (currentTask.current_round > prevTask.current_round) {
                const metricName = currentTask.metric.toUpperCase();
                const latestMetric = currentTask.metric_history[currentTask.metric_history.length - 1];
                logEvent(`Task '${taskId}' round ${prevTask.current_round} complete. ${metricName}: ${latestMetric.toFixed(2)}`);
            }
            // Log status change
            if (currentTask.status !== prevTask.status) {
                 logEvent(`Task '${taskId}' status changed to: ${currentTask.status}`);
            }

            // --- TRIGGER PULSES ---
            // If we are viewing the current task, check for new clients that finished training
            if (state.selectedTaskId === taskId && state.globe) {
                const prevReadyClients = prevTask.selected_clients || [];
                const currentReadyClients = currentTask.selected_clients || [];
                
                // Find clients that are in the current list but were not in the previous one
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

    // --- MAIN DATA FETCHING LOOP ---
    async function fetchData() {
        try {
            const [statusResponse, tokenomicsResponse] = await Promise.all([
                fetch('/status'),
                fetch('/tokenomics')
            ]);

            if (!statusResponse.ok || !tokenomicsResponse.ok) {
                throw new Error('Network response was not ok');
            }
            
            const statusData = await statusResponse.json();
            const tokenomicsData = await tokenomicsResponse.json();
            
            state._prevState = JSON.parse(JSON.stringify({ 
                tasks: state.tasks, 
                network: state.network,
                tokenomics: state.tokenomics
            }));
            
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
    
    // --- EVENT LISTENERS ---
    taskSelect.addEventListener('change', (e) => {
        state.selectedTaskId = e.target.value;
        // When task changes, all arcs need to point to the new server.
        // We clear them here, and they will be rebuilt on the next data fetch.
        if (state.globe) {
            state.globe.clearAllArcs();
            updateGlobeArcs();
        }
        updateTaskDetails();
        updateAccuracyChart();
    });

    function initializeDashboard() {
        logEvent('Dashboard Initialized. Connecting to server...');
        if (typeof Chart === 'undefined' || typeof THREE === 'undefined' || typeof createGlobe === 'undefined') {
            logEvent('Error: A required library (Chart.js, Three.js, or globe.js) failed to load.', 'error');
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
        if (document.visibilityState === 'visible') {
            startPolling();
        } else {
            stopPolling();
        }
    });
    
    initializeDashboard();
});