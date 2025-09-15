// static/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let state = {
        selectedTaskId: null,
        tasks: {},
        network: {},
        tokenomics: {},
        charts: { accuracyChart: null },
        globe: null,
        // --- NEW: STORE PREVIOUS STATE FOR COMPARISON ---
        _prevState: {} 
    };


    // --- DOM ELEMENT REFERENCES ---
    const taskSelect = document.getElementById('task-select');
    const eventLog = document.getElementById('event-log');
    // ... (add other element references here as needed)

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
            data: {
                labels: [],
                datasets: [{
                    label: 'Metric',
                    data: [],
                    borderColor: '#8c7ae6',
                    backgroundColor: 'rgba(140, 122, 230, 0.2)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: false },
                    x: { title: { display: true, text: 'Round' } }
                }
            }
        });
    }
    function render() {
        if (!state.tasks || Object.keys(state.tasks).length === 0) {
            // Don't render if we have no task data yet
            return;
        }
        updateTaskSelector();
        updateTaskDetails();
        updateAccuracyChart();
        updateNetworkEconomics();
        updateGlobe(); // Replaces updateNetworkStatus
    }
    
    // --- CONSOLIDATE GLOBE/NETWORK UPDATES ---
    function updateGlobe() {
        if (!state.globe) return;
        const task = state.tasks[state.selectedTaskId];
        const serverLocation = task ? task.server_location : null;
        
        const connectedClients = state.network.connected_clients || [];
        state.globe.updateClientAndServerPoints(connectedClients, serverLocation);
        
        if (task) {
            const activeArcClientIds = task.selected_clients || [];
            const allClientIds = connectedClients.map(c => c.id);

            activeArcClientIds.forEach(clientId => {
                const client = connectedClients.find(c => c.id === clientId);
                if (client && client.location && serverLocation) {
                    state.globe.addOrUpdateArc(client.id, client.location, serverLocation);
                }
            });

            allClientIds.forEach(clientId => {
                if (!activeArcClientIds.includes(clientId)) {
                    state.globe.removeArc(clientId);
                }
            });
        }
    }
    // --- UI UPDATE FUNCTIONS ---
    function updateTaskSelector() {
        // Clear previous options
        taskSelect.innerHTML = '';
        Object.keys(state.tasks).forEach(taskId => {
            const option = document.createElement('option');
            option.value = taskId;
            option.textContent = taskId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); // Nicer name
            taskSelect.appendChild(option);
        });
        // If no task is selected, select the first one
        if (!state.selectedTaskId && Object.keys(state.tasks).length > 0) {
            state.selectedTaskId = Object.keys(state.tasks)[0];
        }
        taskSelect.value = state.selectedTaskId;
    }
    function updateTaskDetails() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) return;
        
        const task = state.tasks[state.selectedTaskId];

        // --- FIX: CHECK FOR TASK EXISTENCE AND MATCH API RESPONSE ---
        // The /status API puts model_name directly in the task object, not in a 'config' sub-object.
        // Let's add checks to prevent errors if the task data hasn't arrived yet.
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
        chart.data.labels = history.map((_, i) => i); // Rounds 0, 1, 2...
        chart.data.datasets[0].label = metricLabel;
        chart.data.datasets[0].data = history;
        chart.options.scales.y.beginAtZero = (metricLabel !== 'RMSE'); // Don't start RMSE at 0
        
        chart.update();
    }
    function updateNetworkEconomics() {
        // Update client count from network info
        document.getElementById('connected-clients-count').textContent = state.network.connected_clients_count || 0;

        // Update leaderboard and total stake from tokenomics
        const leaderboardBody = document.querySelector("#leaderboard-table tbody");
        leaderboardBody.innerHTML = ''; // Clear table
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
    function generateLiveLogs(prevState, currentState) {
        if (!prevState.tasks) return; // Don't log on the very first fetch

        // Log task status changes
        for (const taskId in currentState.tasks) {
            const prevTask = prevState.tasks[taskId] || {};
            const currentTask = currentState.tasks[taskId];

            if (currentTask.current_round > prevTask.current_round) {
                const metricName = currentTask.metric.toUpperCase();
                const latestMetric = currentTask.metric_history[currentTask.metric_history.length - 1];
                logEvent(`Task '${taskId}' round ${prevTask.current_round} complete. ${metricName}: ${latestMetric.toFixed(2)}`);
            }
            if (currentTask.status !== prevTask.status) {
                 logEvent(`Task '${taskId}' status changed to: ${currentTask.status}`);
            }
        }

        // Log client connections/disconnections
        const prevClients = prevState.network.connected_client_ids || [];
        const currentClients = currentState.network.connected_client_ids || [];
        
        const connected = currentClients.filter(id => !prevClients.includes(id));
        const disconnected = prevClients.filter(id => !currentClients.includes(id));

        connected.forEach(id => logEvent(`Client #${id} connected.`));
        disconnected.forEach(id => logEvent(`Client #${id} disconnected.`, 'warn'));
    }


    // --- MAIN DATA FETCHING LOOP ---
    async function fetchData() {
        try {
            const statusResponse = await fetch('/status');
            const tokenomicsResponse = await fetch('/tokenomics');

            if (!statusResponse.ok || !tokenomicsResponse.ok) {
                throw new Error('Network response was not ok');
            }
            
            const statusData = await statusResponse.json();
            const tokenomicsData = await tokenomicsResponse.json();
            
            generateLiveLogs(state, { tasks: statusData.tasks, network: statusData.network_info });
            state._prevState = JSON.parse(JSON.stringify({ tasks: state.tasks, network: state.network }));
            
            state.tasks = statusData.tasks;
            state.network = statusData.network_info;
            state.tokenomics = tokenomicsData;
            
            // --- CALL THE SINGLE RENDER FUNCTION ---
            render(); 
            
            // ... connection indicator logic is the same ...
        } catch (error) {
            // ... error handling is the same ...
        }
    }
    
    // --- EVENT LISTENERS (MODIFIED) ---
    taskSelect.addEventListener('change', (e) => {
        if (state.globe) {
            state.globe.clearAllArcs();
        }
        state.selectedTaskId = e.target.value;
        render(); // Just call render, it will handle everything
    });

function initializeDashboard() {
        logEvent('Dashboard Initialized. Connecting to server...');

        // --- ROBUSTNESS CHECK: Ensure libraries are loaded ---
        if (typeof Chart === 'undefined') {
            logEvent('Error: Chart.js library not loaded.', 'error');
            console.error('Chart.js is not loaded. Cannot initialize charts.');
            return;
        }
        if (typeof THREE === 'undefined') {
            logEvent('Error: Three.js library not loaded.', 'error');
            console.error('Three.js is not loaded. Cannot initialize globe.');
            return;
        }
        if (typeof createGlobe === 'undefined') {
            logEvent('Error: globe.js module not loaded.', 'error');
            console.error('globe.js is not loaded. Cannot initialize globe.');
            return;
        }
        // ---------------------------------------------------

        initializeAccuracyChart();
        
        const globeContainer = document.getElementById('globe-container');
        if (globeContainer) {
            state.globe = createGlobe(globeContainer);
        }
        
        // Start polling only if the tab is initially visible
        if (document.visibilityState === 'visible') {
            startPolling();
        }
    }

    let fetchDataInterval;

    function startPolling() {
        if (fetchDataInterval) clearInterval(fetchDataInterval);
        fetchData();
        fetchDataInterval = setInterval(fetchData, 3000);
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