document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let state = {
        selectedTaskId: null, tasks: {}, network: {}, tokenomics: {},
        charts: { accuracyChart: null }, globe: null, _prevState: {} 
    };

    // --- DOM ELEMENT REFERENCES ---
    const sidebarNav = document.querySelector('.sidebar-nav');
    const mainContent = document.querySelector('.main-content');
    const eventLog = document.getElementById('event-log');
    const connectionStatusDot = document.getElementById('connection-status-dot');
    const connectionStatusText = document.getElementById('connection-status-text');
    const taskSelectContainer = document.getElementById('task-select-container');
    const selectedTaskDisplay = document.getElementById('selected-task-display');
    const taskOptionsList = document.getElementById('task-options-list');
    const economicsTabs = document.querySelectorAll('.tab');
    const economicsTabContents = document.querySelectorAll('.tab-content');
    const taskBuilderForm = document.getElementById('task-builder-form');
    const marketplaceGrid = document.getElementById('marketplace-grid');
    const rotationToggle = document.getElementById('toggle-rotation');
    const cloudsToggle = document.getElementById('toggle-clouds');
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
            data: { labels: [], datasets: [{ label: 'Metric', data: [], borderColor: '#8A3FFC', backgroundColor: 'rgba(138, 63, 252, 0.2)', fill: true, tension: 0.3 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: false, ticks: { color: '#8A93A2' } }, x: { title: { display: true, text: 'Round/Aggregation' }, ticks: { color: '#8A93A2' } } }, plugins: { legend: { labels: { color: '#D1D5DB' } } } }
        });
    }

    // --- SMART RENDER FUNCTION ---
    function render() {
        const prev = state._prevState;
        
        if (JSON.stringify(prev.tasks) !== JSON.stringify(state.tasks)) {
            updateTaskSelector();
            updateTaskDetails();
            updateAccuracyChart();
            updateLiveAccuracy();
            updateGlobeArcs();
            updateMarketplace(); // Update marketplace when tasks change
            generateLiveLogsAndPulses(prev, state);
        }
        if (JSON.stringify(prev.network) !== JSON.stringify(state.network)) {
            updateGlobePointsAndArcs();
        }
        if (JSON.stringify(prev.tokenomics) !== JSON.stringify(state.tokenomics)) {
            updateNetworkEconomics();
        }
    }
    
    function updateGlobePointsAndArcs() {
        if (!state.globe) return;
        const connectedClients = state.network.connected_clients || [];
        // --- MODIFIED: Pass tokenomics data to the globe ---
        state.globe.updateClientPoints(connectedClients, state.tokenomics);
        updateGlobeArcs();
    }

    function updateGlobeArcs() {
        if (!state.globe) return;
        const task = state.tasks[state.selectedTaskId];
        const serverLocation = task ? task.server_location : null;
        state.globe.updateServerPoint(serverLocation);
        const connectedClients = state.network.connected_clients || [];
        connectedClients.forEach(client => {
            if (client.location && serverLocation) {
                state.globe.addOrUpdateArc(client.id, client.location, serverLocation);
            }
        });
        state.globe.removeInactiveArcs(connectedClients.map(c => c.id));
    }

    // --- UI UPDATE FUNCTIONS ---
    function updateTaskSelector() {
        const currentTaskIds = Object.keys(state.tasks);
        if (currentTaskIds.length === 0) return;
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) {
            state.selectedTaskId = currentTaskIds[0];
        }
        const selectedTaskName = state.selectedTaskId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        selectedTaskDisplay.querySelector('span').textContent = selectedTaskName;
        taskOptionsList.innerHTML = '';
        currentTaskIds.forEach(taskId => {
            const option = document.createElement('div');
            option.className = 'task-option';
            if (taskId === state.selectedTaskId) option.classList.add('selected');
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
        document.getElementById('task-progress').textContent = `${task.current_round || 0} / ${task.total_rounds || 0} ${task.learning_mode === 'asynchronous' ? 'Aggs' : 'Rounds'}`;
    }
    
    function updateLiveAccuracy() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) return;
        const task = state.tasks[state.selectedTaskId];
        const history = task.metric_history;
        document.getElementById('latest-accuracy-label').textContent = `Latest ${task.metric.toUpperCase()}`;
        document.getElementById('latest-accuracy').textContent = (history && history.length > 0) ? history[history.length - 1].toFixed(2) : '--';
    }

    function updateAccuracyChart() {
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId] || !state.charts.accuracyChart) return;
        const task = state.tasks[state.selectedTaskId];
        const chart = state.charts.accuracyChart;
        chart.data.labels = task.metric_history.map((_, i) => i);
        chart.data.datasets[0].label = task.metric.toUpperCase();
        chart.data.datasets[0].data = task.metric_history;
        chart.options.scales.y.beginAtZero = (task.metric !== 'rmse');
        chart.update();
    }

    function updateNetworkEconomics() {
        document.getElementById('connected-clients-count').textContent = state.network.connected_clients?.length || 0;
        const globalLeaderboardBody = document.querySelector("#leaderboard-table-global tbody");
        globalLeaderboardBody.innerHTML = '';
        let totalStake = 0;
        if (state.tokenomics) {
            Object.entries(state.tokenomics).forEach(([clientId, account]) => {
                globalLeaderboardBody.insertRow().innerHTML = `<td>${clientId}</td><td>${account.stake.toFixed(2)}</td><td>${account.balance.toFixed(2)}</td>`;
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
                logEvent(`Task '${taskId}' round ${currentTask.current_round} complete. ${metricName}: ${latestMetric.toFixed(2)}`);

                // --- NEW: Trigger server glow and broadcast pulses on aggregation ---
                if (state.globe && taskId === state.selectedTaskId) {
                    state.globe.triggerServerGlow();
                    const participatingClients = currentTask.selected_clients || [];
                    
                    participatingClients.forEach((clientId, index) => {
                        // Add a small delay to create a "wave" effect
                        setTimeout(() => {
                            state.globe.triggerBroadcastPulse(clientId);
                        }, index * 100); 
                    });
                }
                // ----------------------------------------------------------------------
            }
            if (currentTask.status !== prevTask.status) {
                logEvent(`Task '${taskId}' status changed to: ${currentTask.status}`);
            }

            if (state.selectedTaskId === taskId && state.globe) {
                const newlyReadyClients = (currentTask.selected_clients || []).filter(id => !(prevTask.selected_clients || []).includes(id));
                
                newlyReadyClients.forEach(clientId => {
                    logEvent(`Client #${clientId} finished training for task '${taskId}'.`, 'success');
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

    // --- NEW: MARKETPLACE UI UPDATE ---
    function updateMarketplace() {
        marketplaceGrid.innerHTML = '';
        if (!state.tasks || Object.keys(state.tasks).length === 0) {
            marketplaceGrid.innerHTML = '<p>No active federations in the marketplace.</p>';
            return;
        }
        Object.values(state.tasks).forEach(task => {
            const isCompleted = task.status === 'COMPLETED';
            const latestMetric = task.metric_history.length > 0 ? task.metric_history[task.metric_history.length - 1].toFixed(2) : 'N/A';
            const cardHTML = `
                <div class="bounty-card">
                    <div class="bounty-header">
                        <h3 class="bounty-title">${task.task_id.replace(/_/g, ' ')}</h3>
                        <span class="bounty-status ${isCompleted ? 'completed' : 'active'}">${isCompleted ? 'Completed' : 'Active'}</span>
                    </div>
                    <div class="bounty-details">
                        <p><strong>Dataset:</strong> <span>${task.dataset_name}</span></p>
                        <p><strong>Model:</strong> <span>${task.model_name}</span></p>
                        <p><strong>Mode:</strong> <span>${task.learning_mode}</span></p>
                        <p><strong>Progress:</strong> <span>${task.current_round}/${task.total_rounds}</span></p>
                        <p><strong>Metric (${task.metric.toUpperCase()}):</strong> <span>${latestMetric}</span></p>
                        <p><strong>Privacy:</strong> <span>${task.privacy_profile.toUpperCase().replace('_', ' + ')}</span></p>
                    </div>
                    <div class="bounty-footer">
                        <div class="bounty-value">1,500 PHOENIX</div>
                        <div class="bounty-label">Total Bounty</div>
                    </div>
                </div>
            `;
            marketplaceGrid.insertAdjacentHTML('beforeend', cardHTML);
        });
    }

    // --- DATA FETCHING ---
    async function fetchData() {
        try {
            const [statusResponse, tokenomicsResponse] = await Promise.all([fetch('/status'), fetch('/tokenomics')]);
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
    
    // --- EVENT LISTENERS ---
    selectedTaskDisplay.addEventListener('click', () => taskSelectContainer.classList.toggle('open'));
    taskOptionsList.addEventListener('click', (e) => {
        const option = e.target.closest('.task-option');
        if (!option) return;
        const newTaskId = option.dataset.value;
        if (newTaskId !== state.selectedTaskId) {
            state.selectedTaskId = newTaskId;
            
            // --- NEW: Animate camera on task change ---
            const newTask = state.tasks[newTaskId];
            if (state.globe && newTask && newTask.server_location) {
                state.globe.flyTo(newTask.server_location);
            }
            // -------------------------------------------

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
    window.addEventListener('click', (e) => { if (!taskSelectContainer.contains(e.target)) taskSelectContainer.classList.remove('open'); });
    economicsTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            economicsTabs.forEach(t => t.classList.remove('active'));
            economicsTabContents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-content-${tab.dataset.tab}`).classList.add('active');
        });
    });
    rotationToggle.addEventListener('change', (e) => {
        if (state.globe) {
            state.globe.toggleRotation(e.target.checked);
        }
    });

    cloudsToggle.addEventListener('change', (e) => {
        if (state.globe) {
            state.globe.toggleClouds(e.target.checked);
        }
    });
    // --- NEW: VIEW SWITCHING LISTENER ---
    sidebarNav.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (!link || !link.dataset.view) return;
        e.preventDefault();
        
        // Update active link
        sidebarNav.querySelector('a.active').classList.remove('active');
        link.classList.add('active');

        // Update active view
        mainContent.querySelector('.view-container.active-view').classList.remove('active-view');
        document.getElementById(link.dataset.view).classList.add('active-view');
    });

    // --- NEW: FEDERATION BUILDER FORM SUBMISSION ---
    taskBuilderForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(taskBuilderForm);
        const config = {};
        formData.forEach((value, key) => {
            // Convert numerical strings to numbers
            config[key] = isNaN(value) || value === '' ? value : Number(value);
        });

        logEvent(`Attempting to create new task: '${config.task_id}'...`);

        try {
            const response = await fetch('/create-task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Unknown error');
            }
            logEvent(`Successfully created task '${config.task_id}'!`, 'success');
            taskBuilderForm.reset();
        } catch (error) {
            logEvent(`Failed to create task: ${error.message}`, 'error');
        }
    });

    // --- INITIALIZATION ---
    function initializeDashboard() {
        logEvent('Dashboard Initialized. Connecting to server...');
        if (typeof Chart === 'undefined' || typeof THREE === 'undefined' || typeof createGlobe === 'undefined') {
            logEvent('Error: A required library failed to load.', 'error'); return;
        }
        initializeAccuracyChart();
        const globeContainer = document.getElementById('globe-container');
        if (globeContainer) state.globe = createGlobe(globeContainer);
        if (document.visibilityState === 'visible') startPolling();
    }

    let fetchDataInterval;
    function startPolling() { if (fetchDataInterval) clearInterval(fetchDataInterval); fetchData(); fetchDataInterval = setInterval(fetchData, 2000); } // Faster polling
    function stopPolling() { clearInterval(fetchDataInterval); }
    document.addEventListener('visibilitychange', () => document.visibilityState === 'visible' ? startPolling() : stopPolling());
    
    initializeDashboard();
});