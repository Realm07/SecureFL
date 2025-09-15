// In controller.js, this file requires significant changes to its state management.
// Replace the entire file with this new version.

document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = window.location.origin;
    const state = {
        clientId: null,
        sessionToken: null,
        isAutoLooping: false,
        statusPollInterval: null,
    };

    // --- DOM ELEMENT REFERENCES ---
    const screens = { login: document.getElementById('login-screen'), waiting: document.getElementById('waiting-screen'), control: document.getElementById('control-screen') };
    const steps = { data: document.getElementById('step-data'), actions: document.getElementById('step-actions'), rewarded: document.getElementById('step-rewarded') };
    const connectionStatus = document.getElementById('connection-status');
    const slotContainer = document.getElementById('slot-container');
    const dataGrid = document.getElementById('data-grid');
    const confirmDataBtn = document.getElementById('confirm-data-btn');
    const actionBtns = document.querySelectorAll('.action-btn');
    const nextRoundBtn = document.getElementById('next-round-btn');
    const autoLoopToggle = document.getElementById('auto-loop');

    // --- UI HELPER FUNCTIONS ---
    function showScreen(screenName) {
        Object.values(screens).forEach(s => s.classList.remove('active'));
        screens[screenName].classList.add('active');
    }
    function showStep(stepName) {
        Object.values(steps).forEach(s => s.classList.remove('active'));
        steps[stepName].classList.add('active');
    }
    function updateConnectionStatus(isConnected) {
        connectionStatus.className = `status-dot ${isConnected ? 'connected' : 'disconnected'}`;
    }

    // --- API COMMUNICATION ---
    async function apiCall(endpoint, method = 'GET', body = null) {
        try {
            const options = { method, headers: { 'Content-Type': 'application/json' } };
            if (body) options.body = JSON.stringify(body);
            const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
            updateConnectionStatus(true);
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'API request failed');
            }
            return response.status === 200 ? await response.json() : {};
        } catch (error) {
            updateConnectionStatus(false);
            console.error(`API Error on ${endpoint}:`, error);
            alert(`Connection Error: ${error.message}`);
            throw error;
        }
    }

    // --- APPLICATION LOGIC ---
    async function initializeLogin() {
        try {
            const { available } = await apiCall('/controller/slots');
            slotContainer.innerHTML = '';
            available.forEach(slotId => {
                const btn = document.createElement('button');
                btn.className = 'slot-btn';
                btn.textContent = `Client ${slotId}`;
                btn.onclick = () => joinFederation(slotId);
                slotContainer.appendChild(btn);
            });
        } catch (e) { /* Handled in apiCall */ }
    }

    async function joinFederation(clientId) {
        try {
            const data = await apiCall('/controller/join', 'POST', { client_id: clientId });
            state.clientId = data.client_id;
            state.sessionToken = data.session_token;
            document.getElementById('client-id-display').textContent = state.clientId;
            document.getElementById('client-id-display-2').textContent = state.clientId;
            showScreen('waiting');
            startStatusPolling();
        } catch (e) {
            document.getElementById('login-status').textContent = `Error: ${e.message}`;
        }
    }
    
    async function pollStatus() {
        if (!state.clientId || !state.sessionToken) return;
        try {
            const status = await apiCall('/controller/status', 'POST', {
                client_id: state.clientId,
                session_token: state.sessionToken
            });

            if (status.current_step === 'control_panel') {
                stopStatusPolling();
                showScreen('control');
                setupControlScreenForTask(status.task_info);
            }
        } catch (error) {
            console.error("Status poll failed:", error);
            stopStatusPolling();
        }
    }

    function startStatusPolling() {
        if (state.statusPollInterval) clearInterval(state.statusPollInterval);
        state.statusPollInterval = setInterval(pollStatus, 3000);
    }

    function stopStatusPolling() {
        if (state.statusPollInterval) clearInterval(state.statusPollInterval);
        state.statusPollInterval = null;
    }

    function setupControlScreenForTask(taskInfo) {
        document.getElementById('task-title').textContent = `Task: ${taskInfo.task_id}`;
        resetActionButtons();
        populateDataGrid();
        const dpButton = document.querySelector('.action-btn[data-action="dp"]');
        dpButton.style.display = taskInfo.privacy_profile.includes('dp') ? 'flex' : 'none';
        showStep('data');
    }

    function populateDataGrid() {
        dataGrid.innerHTML = '';
        for (let i = 0; i < 16; i++) {
            const snippet = document.createElement('div');
            snippet.className = 'data-snippet';
            snippet.textContent = `[${(Math.random()*2-1).toFixed(2)}, ...]`;
            snippet.onclick = () => {
                snippet.classList.toggle('selected');
                confirmDataBtn.disabled = dataGrid.querySelectorAll('.selected').length === 0;
            };
            dataGrid.appendChild(snippet);
        }
    }
    
    confirmDataBtn.addEventListener('click', () => {
        showStep('actions');
        document.querySelector('.action-btn[data-action="stake"]').classList.add('enabled');
    });

    function resetActionButtons() {
        actionBtns.forEach((btn, index) => {
            btn.classList.remove('enabled', 'completed');
            btn.disabled = false;
            const action = btn.dataset.action;
            const icon = btn.querySelector('i').className;
            const number = btn.querySelector('span').textContent;
            const actionText = action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            btn.innerHTML = `<span>${number}</span> <i class="${icon}"></i> ${actionText}`;
        });
    }
    
    actionBtns.forEach((btn, index) => {
    btn.addEventListener('click', async () => {
        if (!btn.classList.contains('enabled')) return;
        
        const action = btn.dataset.action;
        const originalHTML = btn.innerHTML;
        btn.classList.remove('enabled');
        btn.innerHTML = `<span>${index+1}.</span> <i class="fas fa-spinner fa-spin"></i> Processing...`;

        try {
            // The server now tells us exactly what the next step is
            const response = await apiCall('/controller/action', 'POST', {
                client_id: state.clientId,
                session_token: state.sessionToken,
                action: action
            });
            
            await new Promise(res => setTimeout(res, 750 + Math.random() * 500));
            
            btn.classList.add('completed');
            btn.innerHTML = `<span>${index+1}.</span> <i class="fas fa-check"></i> ${action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} Complete`;

            // --- NEW LOGIC ---
            // Use the server's response to determine what to do next
            if (response.next_step === "rewarded") {
                showStep('rewarded');
            } else {
                // Find the next button to enable based on its data-action attribute
                const nextAction = response.next_step.replace('actions_', ''); // e.g., "actions_train" -> "train"
                const nextButton = document.querySelector(`.action-btn[data-action="${nextAction}"]`);
                if (nextButton) {
                    nextButton.classList.add('enabled');
                }
            }
            // --- END NEW LOGIC ---

        } catch (error) {
            alert(`Action failed: ${error.message}`);
            btn.innerHTML = originalHTML; // Restore button on error
            btn.classList.add('enabled');
        }
    });
});
    nextRoundBtn.addEventListener('click', () => {
        showScreen('waiting');
        startStatusPolling();
    });
    
    autoLoopToggle.addEventListener('change', (e) => { state.isAutoLooping = e.target.checked; });

    // --- INITIALIZATION ---
    showScreen('login');
    initializeLogin();
});