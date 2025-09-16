document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = window.location.origin;
    const state = { clientId: null, sessionToken: null, statusPollInterval: null };

    // DOM References
    const screens = { login: document.getElementById('login-screen'), waiting: document.getElementById('waiting-screen'), control: document.getElementById('control-screen') };
    const steps = { data: document.getElementById('step-data'), actions: document.getElementById('step-actions'), rewarded: document.getElementById('step-rewarded') };
    const connectionStatus = document.getElementById('connection-status');
    const slotContainer = document.getElementById('slot-container');
    const dataGrid = document.getElementById('data-grid');
    const confirmDataBtn = document.getElementById('confirm-data-btn');
    const actionBtns = document.querySelectorAll('.action-btn');
    const nextRoundBtn = document.getElementById('next-round-btn');

    // UI Helpers
    function showScreen(screenName) { Object.values(screens).forEach(s => s.classList.remove('active')); screens[screenName].classList.add('active'); }
    function showStep(stepName) { Object.values(steps).forEach(s => s.classList.remove('active')); steps[stepName].classList.add('active'); }
    function updateConnectionStatus(isConnected) { connectionStatus.className = `status-dot ${isConnected ? 'connected' : 'disconnected'}`; }

    // API Call Wrapper
    async function apiCall(endpoint, method = 'GET', body = null) {
        try {
            const options = { method, headers: { 'Content-Type': 'application/json' } };
            if (body) options.body = JSON.stringify(body);
            const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
            updateConnectionStatus(true);
            if (!response.ok) throw new Error((await response.json()).error || 'API request failed');
            return response.status === 200 ? await response.json() : {};
        } catch (error) {
            updateConnectionStatus(false);
            alert(`Connection Error: ${error.message}`);
            throw error;
        }
    }

    // Main Application Flow
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
        } catch (e) { console.error("Could not fetch slots:", e); }
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
        } catch (e) { document.getElementById('login-status').textContent = `Error: ${e.message}`; }
    }
    
    async function pollStatus() {
        if (!state.clientId) return;
        try {
            const status = await apiCall('/controller/status', 'POST', {
                client_id: state.clientId, session_token: state.sessionToken
            });
            // --- FIX: Transition to control panel only when the server says so ---
            if (status.current_step.startsWith('control_panel')) {
                stopStatusPolling();
                showScreen('control');
                setupControlScreenForTask(status);
            } else if (status.current_step === 'rewarded') {
                stopStatusPolling();
                showScreen('control');
                showStep('rewarded');
            }
        } catch (error) { console.error("Status poll failed:", error); }
    }

    function startStatusPolling() {
        if (state.statusPollInterval) clearInterval(state.statusPollInterval);
        state.statusPollInterval = setInterval(pollStatus, 3000);
    }
    function stopStatusPolling() { clearInterval(state.statusPollInterval); state.statusPollInterval = null; }

    function setupControlScreenForTask(status) {
        document.getElementById('task-title').textContent = `Task: ${status.task_info.task_id}`;
        resetActionButtons();
        populateDataGrid();
        const dpButton = document.querySelector('.action-btn[data-action="dp"]');
        dpButton.style.display = status.task_info.privacy_profile.includes('dp') ? 'flex' : 'none';
        
        const initialStep = status.current_step.replace('control_panel_', '');
        showStep(initialStep === 'data' ? 'data' : 'actions');
        
        if(initialStep !== 'data') {
            const action = status.current_step.replace('actions_', '');
            const btnToEnable = document.querySelector(`.action-btn[data-action="${action}"]`);
            if(btnToEnable) btnToEnable.classList.add('enabled');
        } else {
             // Default start: enable nothing until data is confirmed
        }
    }

    function populateDataGrid() {
        dataGrid.innerHTML = '';
        confirmDataBtn.disabled = true;
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
    
    confirmDataBtn.addEventListener('click', () => handleAction('confirm_data'));

    function resetActionButtons() {
        actionBtns.forEach(btn => {
            btn.classList.remove('enabled', 'completed');
            const action = btn.dataset.action;
            const icon = btn.querySelector('i').className;
            const number = btn.querySelector('span').textContent;
            btn.innerHTML = `<span>${number}</span> <i class="${icon}"></i> ${action.charAt(0).toUpperCase() + action.slice(1)}`;
        });
    }

    async function handleAction(action, btnElement = null) {
        if (btnElement && !btnElement.classList.contains('enabled')) return;
        
        if (btnElement) {
            btnElement.classList.remove('enabled');
            btnElement.innerHTML = `<span>${btnElement.querySelector('span').textContent}</span> <i class="fas fa-spinner fa-spin"></i> Processing...`;
        }

        try {
            const response = await apiCall('/controller/action', 'POST', {
                client_id: state.clientId, session_token: state.sessionToken, action: action
            });
            
            if (btnElement) {
                await new Promise(res => setTimeout(res, 750));
                btnElement.classList.add('completed');
                const actionText = action.charAt(0).toUpperCase() + action.slice(1);
                btnElement.innerHTML = `<span>${btnElement.querySelector('span').textContent}</span> <i class="fas fa-check"></i> ${actionText} Complete`;
            }

            const nextStepAction = response.next_step.replace('actions_', '');
            
            // --- FIX: Correctly handle the new waiting_for_aggregation state ---
            if (response.next_step === "waiting_for_aggregation") {
                showStep('waiting_agg');
                startStatusPolling(); // Start polling to see when we get rewarded
            } else if (response.next_step.startsWith('actions_')) {
                showStep('actions');
                const nextButton = document.querySelector(`.action-btn[data-action="${nextStepAction}"]`);
                if (nextButton) nextButton.classList.add('enabled');
            }

        } catch (error) {
            if (btnElement) btnElement.classList.add('enabled');
        }
    }


    actionBtns.forEach(btn => btn.addEventListener('click', () => handleAction(btn.dataset.action, btn)));
    nextRoundBtn.addEventListener('click', () => { showScreen('waiting'); startStatusPolling(); });
    
    initializeLogin();
});