document.addEventListener('DOMContentLoaded', () => {
    let state = {
        selectedTaskId: null, tasks: {}, network: {}, tokenomics: {},
        charts: { accuracyChart: null }, globe: null, _prevState: {} 
    };
    let ledgerState = {
        chain: [],
        currentPage: 0,
        blocksPerPage: 20,
        isLoading: false,
        currentTaskId: null
    };

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
    const ledgerGrid = document.getElementById('ledger-grid');
    const ledgerRefreshBtn = document.getElementById('ledger-refresh-btn');
    const ledgerLoadMoreBtn = document.getElementById('ledger-load-more-btn');
    const ledgerTaskDisplay = document.getElementById('ledger-task-display');
    const clientIdModalOverlay = document.getElementById('client-id-modal-overlay');
    const clientIdForm = document.getElementById('client-id-form');
    const clientIdInput = document.getElementById('client-id-input');
    const modalHeaderText = clientIdModalOverlay.querySelector('.modal-header');
    const modalBodyText = clientIdModalOverlay.querySelector('.modal-text');


    function logEvent(message, type = 'info') {
        const logEntry = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> <span class="log-message">${message}</span>`;
        logEntry.className = `log-entry log-${type}`;
        eventLog.prepend(logEntry);
    }

    function initializeAccuracyChart() {
        const ctx = document.getElementById('accuracy-chart').getContext('2d');
        state.charts.accuracyChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Metric', data: [], borderColor: '#8A3FFC', backgroundColor: 'rgba(138, 63, 252, 0.2)', fill: true, tension: 0.3 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: false, ticks: { color: '#8A93A2' } }, x: { title: { display: true, text: 'Round/Aggregation' }, ticks: { color: '#8A93A2' } } }, plugins: { legend: { labels: { color: '#D1D5DB' } } } }
        });
    }

    function renderLedgerBlock(block) {
        const metricName = state.tasks[ledgerState.currentTaskId]?.metric?.toUpperCase() || 'METRIC';
        const metricValue = block.round_data.global_model_accuracy !== undefined ? block.round_data.global_model_accuracy.toFixed(2) : 'N/A';
        const roundData = block.round_data.message ? `<p><strong>Message:</strong> <span>${block.round_data.message}</span></p>` : `
            <p><strong>Round Number:</strong> <span>${block.round_data.round_number}</span></p>
            <p><strong>Participants:</strong> <span>[${block.round_data.participants.join(', ')}]</span></p>
            <p><strong>${metricName}:</strong> <span>${metricValue}</span></p>
            <p><strong>Model Hash:</strong> <span class="hash-value">${block.round_data.global_model_hash.substring(0, 32)}...</span></p>
        `;

        return `
            <div class="ledger-block-card">
                <div class="ledger-block-header">
                    <span class="block-index">Block #${block.index}</span>
                    <span class="block-timestamp">${new Date(block.timestamp * 1000).toLocaleString()}</span>
                </div>
                <div class="ledger-block-body">
                    ${roundData}
                    <p><strong>Previous Hash:</strong> <span class="hash-value">${block.previous_hash.substring(0, 32)}...</span></p>
                </div>
            </div>
        `;
    }

    function renderLedgerPage() {
        if (ledgerState.isLoading) {
            ledgerGrid.innerHTML = `<p class="placeholder-message">Loading ledger...</p>`;
            ledgerLoadMoreBtn.style.display = 'none';
            return;
        }

        if (ledgerState.chain.length === 0) {
            ledgerGrid.innerHTML = `<p class="placeholder-message">No ledger found for this task, or task not selected. Click Refresh.</p>`;
            ledgerLoadMoreBtn.style.display = 'none';
            return;
        }

        const reversedChain = [...ledgerState.chain].reverse();
        const start = 0;
        const end = (ledgerState.currentPage + 1) * ledgerState.blocksPerPage;
        const blocksToShow = reversedChain.slice(start, end);

        if (ledgerState.currentPage === 0) {
            ledgerGrid.innerHTML = '';
        }
        
        ledgerGrid.innerHTML = blocksToShow.map(renderLedgerBlock).join('');
        
        if (end < reversedChain.length) {
            ledgerLoadMoreBtn.style.display = 'block';
        } else {
            ledgerLoadMoreBtn.style.display = 'none';
        }
    }

    async function fetchAndRenderLedger() {
        if (!state.selectedTaskId) {
            logEvent('Cannot fetch ledger: No task selected.', 'warn');
            return;
        }
        
        ledgerState.isLoading = true;
        ledgerState.currentTaskId = state.selectedTaskId;
        ledgerState.currentPage = 0;
        renderLedgerPage();

        try {
            const response = await fetch(`/tasks/${state.selectedTaskId}/ledger`);
            if (!response.ok) throw new Error(`Network response was not ok (${response.status})`);
            
            const chainData = await response.json();
            ledgerState.chain = Array.isArray(chainData) ? chainData : [];
        } catch (error) {
            console.error("Failed to fetch ledger:", error);
            logEvent(`Failed to fetch ledger for '${state.selectedTaskId}': ${error.message}`, 'error');
            ledgerState.chain = []; 
        } finally {
            ledgerState.isLoading = false;
            renderLedgerPage();
        }
    }

    function prepareLedgerView() {
        const taskName = state.selectedTaskId ? state.selectedTaskId.replace(/_/g, ' ') : 'None';
        ledgerTaskDisplay.innerHTML = `Viewing ledger for: <strong>${taskName}</strong>`;
        
        if (ledgerState.currentTaskId !== state.selectedTaskId) {
            ledgerState.chain = [];
            ledgerState.currentPage = 0;
            ledgerState.currentTaskId = state.selectedTaskId;
            renderLedgerPage(); 
        }
    }

    function render() {
        const prev = state._prevState;
        
        if (JSON.stringify(prev.network) !== JSON.stringify(state.network)) {
            updateGlobePointsAndArcs();
        }
        
        if (JSON.stringify(prev.tasks) !== JSON.stringify(state.tasks)) {
            updateTaskSelector();
            updateTaskDetails();
            updateAccuracyChart();
            updateLiveAccuracy();
            updateGlobeArcs();
            updateMarketplace();
            generateLiveLogsAndPulses(prev, state);
        }
    }
    
    function updateGlobePointsAndArcs() {
        if (!state.globe) return;
        const connectedClients = state.network.connected_clients || [];
        state.globe.updateClientPoints(connectedClients, state.tokenomics);
        updateGlobeArcs();
    }

    function updateGlobeArcs() {
        if (!state.globe) return;
        const task = state.tasks[state.selectedTaskId];
        const serverLocation = task ? task.server_location : null;
        state.globe.updateServerPoint(serverLocation);
        const connectedClients = state.network.connected_clients || [];
        if (serverLocation) {
             connectedClients.forEach(client => {
                if (client.location) {
                    state.globe.addOrUpdateArc(client.id, client.location, serverLocation);
                }
            });
        }
        state.globe.removeInactiveArcs(connectedClients.map(c => c.id));
    }

    function updateTaskSelector() {
        const currentTaskIds = Object.keys(state.tasks);
        if (currentTaskIds.length === 0) return;
        if (!state.selectedTaskId || !state.tasks[state.selectedTaskId]) {
            state.selectedTaskId = currentTaskIds[0];
            prepareLedgerView();
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
        document.getElementById('total-stake').textContent = `${totalStake.toFixed(2)} AFT`;
        updateGlobePointsAndArcs();
    }

    function generateLiveLogsAndPulses(prevState, currentState) {
        if (!prevState.tasks || Object.keys(prevState.tasks).length === 0) return;
        const prevClients = (prevState.network.connected_clients || []).map(c => c.id);
        const currentClients = (currentState.network.connected_clients || []).map(c => c.id);
        currentClients.filter(id => !prevClients.includes(id)).forEach(id => logEvent(`Client #${id} connected.`));
        prevClients.filter(id => !currentClients.includes(id)).forEach(id => logEvent(`Client #${id} disconnected.`, 'warn'));
        
        for (const taskId in currentState.tasks) {
            if (!prevState.tasks[taskId]) continue;
            const prevTask = prevState.tasks[taskId];
            const currentTask = currentState.tasks[taskId];
            if (currentTask.current_round > prevTask.current_round) {
                logEvent(`Task '${taskId}' round ${currentTask.current_round} complete. Metric: ${currentTask.metric_history.slice(-1)[0].toFixed(2)}`);
                if (state.globe && taskId === state.selectedTaskId) {
                    state.globe.triggerServerGlow();
                    const participatingClients = prevTask.selected_clients || [];
                    if (participatingClients.length > 0) {
                        participatingClients.forEach((clientId, index) => {
                            setTimeout(() => { state.globe.triggerBroadcastPulse(clientId); }, index * 100);
                        });
                    }
                }
            }
            const prevSelected = prevTask.selected_clients || [];
            const currentSelected = currentTask.selected_clients || [];
            if (JSON.stringify(prevSelected) !== JSON.stringify(currentSelected)) {
                if (state.selectedTaskId === taskId && state.globe && currentSelected.length > 0) {
                    currentSelected.forEach(clientId => {
                        logEvent(`Client #${clientId} starting training for task '${taskId}'.`, 'success');
                        state.globe.triggerPulse(clientId);
                    });
                }
            }
        }
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

    async function fetchStatusData() {
        try {
            const statusResponse = await fetch('/status');
            if (!statusResponse.ok) throw new Error('Network response was not ok');
            const statusData = await statusResponse.json();
            
            state._prevState = JSON.parse(JSON.stringify({ tasks: state.tasks, network: state.network }));
            state.tasks = statusData.tasks;
            state.network = statusData.network_info;
            
            render();
            connectionStatusDot.className = 'status-dot connected';
            connectionStatusText.textContent = 'Connected';
        } catch (error) {
            connectionStatusDot.className = 'status-dot disconnected';
            connectionStatusText.textContent = 'Disconnected';
            if (!error.message.includes('earthMesh')) {
                console.error("Fetch error:", error);
            }
        }
    }

    async function fetchTokenomicsData() {
        logEvent('Refreshing tokenomics data...');
        try {
            const tokenomicsResponse = await fetch('/tokenomics');
            if (!tokenomicsResponse.ok) throw new Error('Tokenomics fetch failed');
            const tokenomicsData = await tokenomicsResponse.json();
            state.tokenomics = tokenomicsData;
            updateNetworkEconomics();
            logEvent('Tokenomics updated successfully.', 'success');
        } catch (error) {
            logEvent(`Failed to refresh tokenomics: ${error.message}`, 'error');
        }
    }

    function getClientIdWithModal(taskId, amount) {
        return new Promise((resolve, reject) => {
            modalHeaderText.textContent = `Stake on Task: ${taskId.replace(/_/g, ' ')}`;
            modalBodyText.textContent = `You are about to stake ${amount} AFT. Please confirm your Client ID to proceed.`;

            clientIdModalOverlay.classList.add('visible');
            clientIdInput.value = '';
            clientIdInput.focus();

            const handleSubmit = (event) => {
                event.preventDefault();
                const clientId = clientIdInput.value;
                if (clientId.trim() !== '') {
                    cleanupAndResolve(clientId);
                }
            };

            const handleCancel = (event) => {
                if (event.target === clientIdModalOverlay) {
                    cleanupAndReject();
                }
            };
            
            const cleanup = () => {
                clientIdModalOverlay.classList.remove('visible');
                clientIdForm.removeEventListener('submit', handleSubmit);
                clientIdModalOverlay.removeEventListener('click', handleCancel);
            };

            const cleanupAndResolve = (value) => {
                cleanup();
                resolve(value);
            };

            const cleanupAndReject = () => {
                cleanup();
                reject(new Error("User cancelled the operation."));
            };

            clientIdForm.addEventListener('submit', handleSubmit);
            clientIdModalOverlay.addEventListener('click', handleCancel);
        });
    }


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

    tokenomicsRefreshBtn.addEventListener('click', fetchTokenomicsData);

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
            prepareLedgerView();
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
            prepareLedgerView();
        }
    });

    ledgerRefreshBtn.addEventListener('click', fetchAndRenderLedger);
    ledgerLoadMoreBtn.addEventListener('click', () => {
        ledgerState.currentPage++;
        renderLedgerPage();
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

        let clientIdStr;
        try {
            clientIdStr = await getClientIdWithModal(taskId, amount);
        } catch (error) {
            logEvent('Stake operation cancelled.', 'info');
            return;
        }
        
        const clientId = parseInt(clientIdStr, 10);
        if (isNaN(clientId)) {
            logEvent('Invalid Client ID entered.', 'error');
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
            
            logEvent(`Client #${clientId} successfully staked ${amount} AFT!`, 'success');
            input.value = '';
            fetchTokenomicsData();
        } catch (error) {
            logEvent(`Staking failed for Client #${clientId}: ${error.message}`, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Contribute Stake';
        }
    });

    async function initializeDashboard() {
        logEvent('Dashboard Initialized. Connecting to server...');
        if (typeof Chart === 'undefined' || typeof THREE === 'undefined' || typeof createGlobe === 'undefined') {
            logEvent('Error: A required library failed to load.', 'error'); return;
        }
        initializeAccuracyChart();
        const globeContainer = document.getElementById('globe-container');
        if (globeContainer) {
            try {
                state.globe = await createGlobe(globeContainer);
                if (document.visibilityState === 'visible') {
                    startPolling();
                    fetchTokenomicsData();
                }
            } catch (error) {
                logEvent(`FATAL: Could not initialize globe. ${error.message}`, 'error');
            }
        }
    }


    let fetchDataInterval;
    function startPolling() { 
        if (fetchDataInterval) clearInterval(fetchDataInterval); 
        fetchStatusData(); 
        fetchDataInterval = setInterval(fetchStatusData, 5000); 
    }
    function stopPolling() { clearInterval(fetchDataInterval); }
    document.addEventListener('visibilitychange', () => document.visibilityState === 'visible' ? startPolling() : stopPolling());
    window.addEventListener('beforeunload', () => {
        if (state.globe && typeof state.globe.destroy === 'function') {
            state.globe.destroy();
            state.globe = null;
        }
    });
    initializeDashboard();
});