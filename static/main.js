document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let state = {
        selectedTaskId: null, tasks: {}, network: {}, tokenomics: {},
        charts: { accuracyChart: null }, globe: null, _prevState: {} 
    };

    // --- DOM ELEMENT REFERENCES ---
    const sidebar = document.querySelector('.sidebar');
    const sidebarToggleBtn = document.getElementById('sidebar-toggle');
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
    const tokenomicsRefreshBtn = document.getElementById('tokenomics-refresh-btn');
    const taskBuilderForm = document.getElementById('task-builder-form');
    const marketplaceGrid = document.getElementById('marketplace-grid');
    const rotationToggle = document.getElementById('toggle-rotation');
    const cloudsToggle = document.getElementById('toggle-clouds');
    const ledgerContentWrapper = document.getElementById('ledger-content-wrapper');


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
            updateMarketplace();
            if (document.getElementById('ledger-view').classList.contains('active-view')) {
                fetchAndRenderLedger();
            }
            generateLiveLogsAndPulses(prev, state);
        }
        if (JSON.stringify(prev.network) !== JSON.stringify(state.network)) {
            updateGlobePointsAndArcs();
        }
        // --- FIX: Tokenomics is now updated reliably before render ---
        if (JSON.stringify(prev.tokenomics) !== JSON.stringify(state.tokenomics)) {
            updateNetworkEconomics();
        }
    }
    
    function updateGlobePointsAndArcs() {
        if (!state.globe) return;
        const connectedClients = state.network.connected_clients || [];
        // Pass tokenomics data to the globe for tooltips
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
                const clientTotalStake = account.total_stake || 0;
                globalLeaderboardBody.insertRow().innerHTML = `<td>${clientId}</td><td>${clientTotalStake.toFixed(2)}</td><td>${account.balance.toFixed(2)}</td>`;
                totalStake += clientTotalStake;
            });
        }
        document.getElementById('total-stake').textContent = `${totalStake.toFixed(2)} PHOENIX`;
        // Also update globe points as tokenomics data affects tooltips
        updateGlobePointsAndArcs();
    }

    // --- ROBUST: Logic from older script to ensure animations always fire ---
    function generateLiveLogsAndPulses(prevState, currentState) {
        if (!prevState.tasks || Object.keys(prevState.tasks).length === 0 || !state.globe) return;
    
        for (const taskId in currentState.tasks) {
            const prevTask = prevState.tasks[taskId] || { current_round: 0, status: '', metric_history: [], selected_clients: [] };
            const currentTask = currentState.tasks[taskId];
    
            // Check for round completion to trigger animations
            if (currentTask.current_round > prevTask.current_round) {
                const metricName = currentTask.metric.toUpperCase();
                const latestMetric = currentTask.metric_history[currentTask.metric_history.length - 1];
                logEvent(`Task '${taskId}' round ${currentTask.current_round} complete. ${metricName}: ${latestMetric.toFixed(2)}`);
    
                // Trigger animations regardless of which task is selected
                state.globe.triggerServerGlow(currentTask.server_location); 
                const participatingClients = currentTask.selected_clients || [];
                if (Array.isArray(participatingClients)) {
                    participatingClients.forEach((clientId, index) => {
                        setTimeout(() => { 
                            const client = (currentState.network.connected_clients || []).find(c => c.id === clientId);
                            if (client && client.location && currentTask.server_location) {
                                state.globe.triggerBroadcastPulse(client.location, currentTask.server_location); 
                            }
                        }, index * 100); 
                    });
                }
            }
    
            // Log status changes
            if (currentTask.status !== prevTask.status) { 
                logEvent(`Task '${taskId}' status changed to: ${currentTask.status}`); 
            }
    
            // Trigger pulses for newly ready clients (for synchronous tasks)
            const newlyReadyClients = (currentTask.selected_clients || []).filter(id => !(prevTask.selected_clients || []).includes(id));
            newlyReadyClients.forEach(clientId => {
                logEvent(`Client #${clientId} finished training for task '${taskId}'.`, 'success');
                const client = (currentState.network.connected_clients || []).find(c => c.id === clientId);
                if (client && client.location) {
                    state.globe.triggerPulse(client.location);
                }
            });
        }
    
        // Log client connections/disconnections
        const prevClients = (prevState.network.connected_clients || []).map(c => c.id);
        const currentClients = (currentState.network.connected_clients || []).map(c => c.id);
        currentClients.filter(id => !prevClients.includes(id)).forEach(id => logEvent(`Client #${id} connected.`));
        prevClients.filter(id => !currentClients.includes(id)).forEach(id => logEvent(`Client #${id} disconnected.`, 'warn'));
    }

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
                        <input type="number" class="stake-input" placeholder="Amount to Stake" min="1">
                        <button class="button-primary stake-button" data-task-id="${task.task_id}" ${isCompleted ? 'disabled' : ''}>
                            ${isCompleted ? 'Task Complete' : 'Contribute Stake'}
                        </button>
                    </div>
                </div>
            `;
            marketplaceGrid.insertAdjacentHTML('beforeend', cardHTML);
        });
    }

    // --- ROBUST: Combined data fetching from older script ---
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

    async function fetchAndRenderLedger() {
        if (!state.selectedTaskId) {
            ledgerContentWrapper.innerHTML = '<p>Select a task from the Monitor view to see its ledger.</p>';
            return;
        }
        logEvent(`Fetching ledger for task '${state.selectedTaskId}'...`);
        try {
            const response = await fetch(`/tasks/${state.selectedTaskId}/ledger`);
            if (!response.ok) throw new Error('Failed to fetch ledger');
            const ledgerChain = await response.json();
            
            let ledgerHTML = '<div class="ledger-grid">';
            if (!ledgerChain || ledgerChain.length === 0) {
                 ledgerHTML += '<p>No ledger entries found for this task yet.</p>';
            } else {
                [...ledgerChain].reverse().forEach(block => {
                    const timestamp = new Date(block.timestamp * 1000).toLocaleString();
                    const roundData = block.round_data;
                    const isGenesis = roundData.message === 'Genesis Block';
                    
                    ledgerHTML += `
                        <div class="ledger-block-card">
                            <div class="ledger-block-header">
                                <span class="block-index">Block #${block.index}</span>
                                <span class="block-timestamp">${timestamp}</span>
                            </div>
                            <div class="ledger-block-body">
                                ${isGenesis ? `<p><strong>Message:</strong><span>${roundData.message}</span></p>` : `
                                <p><strong>Round:</strong><span>${roundData.round_number}</span></p>
                                <p><strong>Participants:</strong><span>[${roundData.participants.join(', ')}]</span></p>
                                <p><strong>Metric (${state.tasks[state.selectedTaskId]?.metric.toUpperCase() || ''}):</strong><span>${roundData.global_model_accuracy?.toFixed(4) || 'N/A'}</span></p>
                                <p><strong>Model Hash:</strong><span class="hash-value">${roundData.global_model_hash}</span></p>
                                `}
                                <p><strong>Prev. Hash:</strong><span class="hash-value">${block.previous_hash}</span></p>
                            </div>
                        </div>
                    `;
                });
            }
            ledgerHTML += '</div>';
            ledgerContentWrapper.innerHTML = ledgerHTML;

        } catch (error) {
            logEvent(`Error fetching ledger: ${error.message}`, 'error');
            ledgerContentWrapper.innerHTML = `<p>Could not load ledger for ${state.selectedTaskId}.</p>`;
        }
    }
    
    // --- EVENT LISTENERS ---
    sidebarToggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        const icon = sidebarToggleBtn.querySelector('i');
        if (sidebar.classList.contains('collapsed')) {
            icon.classList.remove('fa-angle-double-left');
            icon.classList.add('fa-angle-double-right');
        } else {
            icon.classList.remove('fa-angle-double-right');
            icon.classList.add('fa-angle-double-left');
        }
    });

    tokenomicsRefreshBtn.addEventListener('click', fetchData); // Refresh all data for consistency

    selectedTaskDisplay.addEventListener('click', () => taskSelectContainer.classList.toggle('open'));
    taskOptionsList.addEventListener('click', (e) => {
        const option = e.target.closest('.task-option');
        if (!option) return;
        const newTaskId = option.dataset.value;
        if (newTaskId !== state.selectedTaskId) {
            state.selectedTaskId = newTaskId;
            const newTask = state.tasks[newTaskId];
            if (state.globe && newTask && newTask.server_location) { state.globe.flyTo(newTask.server_location); }
            if (state.globe) { state.globe.clearAllArcs(); updateGlobeArcs(); }
            updateTaskSelector(); updateTaskDetails(); updateAccuracyChart(); updateLiveAccuracy();
            if (document.getElementById('ledger-view').classList.contains('active-view')) {
                fetchAndRenderLedger();
            }
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
    rotationToggle.addEventListener('change', (e) => { if (state.globe) { state.globe.toggleRotation(e.target.checked); } });
    cloudsToggle.addEventListener('change', (e) => { if (state.globe) { state.globe.toggleClouds(e.target.checked); } });
    sidebarNav.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (!link || !link.dataset.view) return;
        e.preventDefault();
        sidebarNav.querySelector('a.active').classList.remove('active');
        link.classList.add('active');
        mainContent.querySelector('.view-container.active-view').classList.remove('active-view');
        document.getElementById(link.dataset.view).classList.add('active-view');

        if (link.dataset.view === 'ledger-view') {
            fetchAndRenderLedger();
        }
    });
    taskBuilderForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(taskBuilderForm);
        const config = {};
        formData.forEach((value, key) => { config[key] = isNaN(value) || value === '' ? value : Number(value); });
        logEvent(`Attempting to create new task: '${config.task_id}'...`);
        try {
            const response = await fetch('/create-task', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) });
            const result = await response.json();
            if (!response.ok) { throw new Error(result.error || 'Unknown error'); }
            logEvent(`Successfully created task '${config.task_id}'!`, 'success');
            taskBuilderForm.reset();
        } catch (error) {
            logEvent(`Failed to create task: ${error.message}`, 'error');
        }
    });

    marketplaceGrid.addEventListener('click', async (e) => {
        if (!e.target.matches('.stake-button')) return;

        const button = e.target;
        const taskId = button.dataset.taskId;
        const input = button.parentElement.querySelector('.stake-input');
        const amount = parseFloat(input.value);

        if (isNaN(amount) || amount <= 0) {
            logEvent('Please enter a valid amount to stake.', 'warn');
            return;
        }

        const clientIdStr = prompt("Enter your Client ID (0-9) to stake tokens:", "0");
        if (clientIdStr === null) return;
        const clientId = parseInt(clientIdStr);
        if (isNaN(clientId)) {
            logEvent('Invalid Client ID.', 'error');
            return;
        }

        logEvent(`Client #${clientId} attempting to stake ${amount} on task '${taskId}'...`);
        button.disabled = true;
        button.textContent = 'Staking...';

        try {
            const response = await fetch('/stake', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_id: clientId, amount, task_id: taskId })
            });
            const result = await response.json();
            if (!response.ok) { throw new Error(result.error || 'Staking failed'); }
            
            logEvent(`Client #${clientId} successfully staked ${amount} PHOENIX!`, 'success');
            input.value = '';
            fetchData(); // Re-fetch all data to update tokenomics and UI
        } catch (error) {
            logEvent(`Staking failed for Client #${clientId}: ${error.message}`, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Contribute Stake';
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
        if (document.visibilityState === 'visible') {
            startPolling();
        }
    }

    let fetchDataInterval;
    function startPolling() { 
        if (fetchDataInterval) clearInterval(fetchDataInterval); 
        fetchData(); 
        fetchDataInterval = setInterval(fetchData, 3000); // Set a reasonable interval
    }
    function stopPolling() { clearInterval(fetchDataInterval); }
    document.addEventListener('visibilitychange', () => document.visibilityState === 'visible' ? startPolling() : stopPolling());
    
    initializeDashboard();
});