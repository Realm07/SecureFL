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

    function updateNetworkStatus() {
        const count = state.network.connected_clients_count || 0;
        document.getElementById('connected-clients-count').textContent = count;
        
        // Update the globe with the list of connected client IDs
        if (state.globe && state.network.connected_client_ids) {
            state.globe.updateClientStatus(state.network.connected_client_ids);
        }
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
            
            // --- NEW: GENERATE LOGS BEFORE UPDATING STATE ---
            generateLiveLogs(state, { tasks: statusData.tasks, network: statusData.network_info });

            // Store a DEEP COPY of the previous state
            state._prevState = JSON.parse(JSON.stringify({ tasks: state.tasks, network: state.network }));
            
            // Update state with new data
            state.tasks = statusData.tasks;
            state.network = statusData.network_info;
            state.tokenomics = tokenomicsData;
            
            // Update connection indicator
            document.getElementById('connection-status-dot').className = 'status-dot connected';
            document.getElementById('connection-status-text').textContent = 'Connected';

        } catch (error) {
            console.error('Failed to fetch data:', error);
            logEvent('Failed to connect to server.', 'error');
            document.getElementById('connection-status-dot').className = 'status-dot disconnected';
            document.getElementById('connection-status-text').textContent = 'Disconnected';
        }
    }
    
    // --- EVENT LISTENERS ---
    taskSelect.addEventListener('change', (e) => {
        state.selectedTaskId = e.target.value;
        // Immediately update UI for the newly selected task
        updateTaskDetails();
        updateAccuracyChart();
    });

    // --- INITIALIZATION ---
    logEvent('Dashboard Initialized. Connecting to server...');
    initializeAccuracyChart();
     const globeContainer = document.getElementById('globe-container');
    if (globeContainer) {
        state.globe = createGlobe(globeContainer);
    }
    let fetchDataInterval;

    function startPolling() {
        // Clear any existing interval to prevent duplicates
        if (fetchDataInterval) clearInterval(fetchDataInterval);
        
        fetchData(); // Fetch immediately
        fetchDataInterval = setInterval(fetchData, 3000); // Poll every 3 seconds
    }

    function stopPolling() {
        clearInterval(fetchDataInterval);
    }

    // --- NEW: HANDLE PAGE VISIBILITY ---
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            console.log('Tab is visible, starting polling.');
            startPolling();
        } else {
            console.log('Tab is hidden, stopping polling.');
            stopPolling();
        }
    });

    // Start polling only if the tab is initially visible
    if (document.visibilityState === 'visible') {
        startPolling();
    }
});