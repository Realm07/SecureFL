document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = window.location.origin; // This will be the ngrok URL
    const state = {
        clientId: null,
        sessionToken: null,
        currentTask: null,
        currentStep: 'login',
        isAutoLooping: false,
    };

    // --- DOM ELEMENT REFERENCES ---
    const screens = {
        login: document.getElementById('login-screen'),
        waiting: document.getElementById('waiting-screen'),
        control: document.getElementById('control-screen'),
    };
    const steps = {
        data: document.getElementById('step-data'),
        actions: document.getElementById('step-actions'),
        rewarded: document.getElementById('step-rewarded'),
    };
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
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'API request failed');
            }
            updateConnectionStatus(true);
            return await response.json();
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
        } catch (e) { /* Error handled in apiCall */ }
    }

    async function joinFederation(clientId) {
        try {
            const data = await apiCall('/controller/join', 'POST', { client_id: clientId });
            state.clientId = data.client_id;
            state.sessionToken = data.session_token;
            document.getElementById('client-id-display').textContent = state.clientId;
            document.getElementById('client-id-display-2').textContent = state.clientId;
            
            // This is a simplified simulation of waiting for a task.
            // In a real system, you'd poll a status endpoint.
            showScreen('control');
            setupControlScreenForTask({ name: 'Arrhythmia Detection' }); // Assume a default task
        } catch (e) {
            document.getElementById('login-status').textContent = `Error: ${e.message}`;
        }
    }

    function setupControlScreenForTask(task) {
        state.currentTask = task;
        document.getElementById('task-title').textContent = `Task: ${task.name}`;
        resetActionButtons();
        populateDataGrid();
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
                const selectedCount = dataGrid.querySelectorAll('.selected').length;
                confirmDataBtn.disabled = selectedCount === 0;
            };
            dataGrid.appendChild(snippet);
        }
    }
    
    confirmDataBtn.addEventListener('click', () => {
        showStep('actions');
        document.querySelector('.action-btn[data-action="stake"]').classList.add('enabled');
    });

    function resetActionButtons() {
        actionBtns.forEach(btn => {
            btn.classList.remove('enabled', 'completed');
            btn.disabled = true;
        });
    }

    actionBtns.forEach((btn, index) => {
        btn.addEventListener('click', async () => {
            if (!btn.classList.contains('enabled')) return;
            
            const action = btn.dataset.action;
            btn.innerHTML = `<span>${index+1}.</span> <i class="fas fa-spinner fa-spin"></i> Processing...`;
            
            // SIMULATE ACTION
            await new Promise(res => setTimeout(res, 1500)); 
            
            btn.classList.remove('enabled');
            btn.classList.add('completed');
            btn.innerHTML = `<span>${index+1}.</span> <i class="fas fa-check"></i> ${action.charAt(0).toUpperCase() + action.slice(1)} Complete`;

            // Enable next button
            if (index < actionBtns.length - 1) {
                actionBtns[index + 1].classList.add('enabled');
            } else {
                // Last action was 'send'
                showStep('rewarded');
            }
        });
    });
    
    nextRoundBtn.addEventListener('click', () => {
        if (state.isAutoLooping) {
            // Logic for automated loop would go here
            alert("Starting automated client loop!");
        }
        showScreen('waiting');
        // Simulate waiting and getting a new task
        setTimeout(() => {
            showScreen('control');
            setupControlScreenForTask({ name: 'NASA Battery Health' });
        }, 5000);
    });

    autoLoopToggle.addEventListener('change', (e) => {
        state.isAutoLooping = e.target.checked;
    });

    // --- INITIALIZATION ---
    showScreen('login');
    initializeLogin();
});