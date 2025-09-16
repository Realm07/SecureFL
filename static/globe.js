// --- CONSTANTS ---
const GLOBE_RADIUS = 100;
const CLIENT_POINT_RADIUS = 0.7;
const SERVER_POINT_RADIUS = 2.0;
const PULSE_CYLINDER_HEIGHT = 5;
const PULSE_MAIN_RADIUS = 0.8;
const PULSE_GLOW_RADIUS = 1.5;
const ARC_THICKNESS = 0.25;

// --- UTILITY FUNCTIONS ---
function latLonToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const z = (radius * Math.sin(phi) * Math.sin(theta));
    const y = (radius * Math.cos(phi));
    return new THREE.Vector3(x, y, z);
}

function createCurve(startVec, endVec) {
    const midPoint = startVec.clone().lerp(endVec, 0.5);
    const distance = startVec.distanceTo(endVec);
    midPoint.normalize().multiplyScalar(GLOBE_RADIUS + distance * 1.75);
    const controlPoint1 = startVec.clone().lerp(midPoint, 0.25);
    const controlPoint2 = endVec.clone().lerp(midPoint, 0.25);
    return new THREE.CubicBezierCurve3(startVec, controlPoint1, controlPoint2, endVec);
}

// --- MAIN GLOBE FUNCTION ---
function createGlobe(container) {
    let scene, camera, renderer, controls, earthMesh, cloudsMesh, composer;
    let clientPoints = new Map();
    let serverPoint = null;
    let serverGlow = null;
    let activeArcs = new Map();
    let activePulses = [];
    const cylinderUp = new THREE.Vector3(0, 1, 0);

    let tooltipElement;
    let raycaster = new THREE.Raycaster();
    let mouse = new THREE.Vector2();
    let currentlyHovered = null;

    // --- FIX 1: Add a handle for the animation loop to allow cancellation ---
    let animationFrameId;

    function init() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
        camera.position.z = 250;

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearAlpha(0.0);
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        tooltipElement = document.createElement('div');
        tooltipElement.className = 'globe-tooltip';
        Object.assign(tooltipElement.style, {
            position: 'absolute', display: 'none', backgroundColor: 'rgba(20, 20, 30, 0.85)',
            color: '#E0E0E0', padding: '8px 12px', borderRadius: '4px', fontFamily: 'sans-serif',
            fontSize: '13px', pointerEvents: 'none', whiteSpace: 'nowrap', zIndex: '100',
            border: '1px solid rgba(138, 63, 252, 0.5)'
        });
        container.appendChild(tooltipElement);

        const title = document.querySelector('.globe-title');
        const subtitle = document.querySelector('.globe-subtitle');
        if(title) title.style.display = 'none';
        if(subtitle) subtitle.style.display = 'none';

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true; controls.dampingFactor = 0.05;
        controls.autoRotate = true; controls.autoRotateSpeed = 0.2;
        controls.enablePan = false; controls.minDistance = 150; controls.maxDistance = 400;

        const textureLoader = new THREE.TextureLoader();
        const earthTexture = textureLoader.load('/static/textures/Earth_Night_Map_HIGH.jpg'); 
        const specularMap = textureLoader.load('/static/textures/Earth_Specular_Map_HIGH.tif'); 
        const earthMaterial = new THREE.MeshPhongMaterial({ map: earthTexture, specularMap: specularMap, specular: new THREE.Color('#111111'), shininess: 5 });
        const earthGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
        earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
        scene.add(earthMesh);

        const cloudTexture = textureLoader.load('/static/textures/Earth_Clouds_HIGH.jpg');
        const cloudMaterial = new THREE.MeshLambertMaterial({ map: cloudTexture, transparent: true, opacity: 0.15 });
        const cloudGeometry = new THREE.SphereGeometry(GLOBE_RADIUS + 0.5, 64, 64);
        cloudsMesh = new THREE.Mesh(cloudGeometry, cloudMaterial);
        scene.add(cloudsMesh);

        const atmosphereMaterial = new THREE.ShaderMaterial({
            vertexShader: `varying vec3 vNormal; void main() { vNormal = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
            fragmentShader: `varying vec3 vNormal; void main() { float intensity = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0); gl_FragColor = vec4(0.0, 0.70, 0.85, 1.0) * intensity; }`,
            blending: THREE.AdditiveBlending, side: THREE.BackSide, transparent: true
        });
        const atmosphereGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.01, 64, 64);
        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        scene.add(atmosphere);

        scene.add(new THREE.HemisphereLight(0xffffff, 0x4A90E2, 0.6));
        scene.add(new THREE.AmbientLight(0xffffff, 0.3));

        composer = new THREE.EffectComposer(renderer);
        composer.addPass(new THREE.RenderPass(scene, camera));
        composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.7, 0.5, 0.85));

        animate();
        window.addEventListener('resize', onWindowResize);
        renderer.domElement.addEventListener('mousemove', onMouseMove);
    }

    // ... (addOrUpdateArc, removeInactiveArcs, clearAllArcs remain the same)
    function addOrUpdateArc(clientId, clientLocation, serverLocation) {
        const key = `arc-${clientId}`;
        if (activeArcs.has(key)) return;
        const startVec = latLonToVector3(clientLocation.lat, clientLocation.lon, GLOBE_RADIUS);
        const endVec = latLonToVector3(serverLocation.lat, serverLocation.lon, GLOBE_RADIUS);
        const curve = createCurve(startVec, endVec);
        const geometry = new THREE.TubeGeometry(curve, 64, ARC_THICKNESS, 8, false);
        const material = new THREE.MeshBasicMaterial({ color: 0x00B4D8, transparent: true, opacity: 0 });
        const arcMesh = new THREE.Mesh(geometry, material);
        earthMesh.add(arcMesh);
        let progress = { value: 0 };
        const tween = new TWEEN.Tween(progress).to({ value: 1 }, 500).onUpdate(() => { arcMesh.material.opacity = progress.value; }).start();
        activeArcs.set(key, { mesh: arcMesh, curve, tween });
    }

    function removeInactiveArcs(connectedClientIds) {
        activeArcs.forEach((arcData, key) => {
            const arcClientId = parseInt(key.split('-')[1]);
            if (!connectedClientIds.includes(arcClientId)) {
                earthMesh.remove(arcData.mesh);
                arcData.mesh.geometry.dispose();
                arcData.mesh.material.dispose();
                TWEEN.remove(arcData.tween);
                activeArcs.delete(key);
            }
        });
    }

    function clearAllArcs() {
        activeArcs.forEach((arcData) => {
            earthMesh.remove(arcData.mesh);
            arcData.mesh.geometry.dispose();
            arcData.mesh.material.dispose();
            TWEEN.remove(arcData.tween);
        });
        activeArcs.clear();
    }
    
    function createPulse(curve, color) {
        const pulseGeom = new THREE.CylinderGeometry(PULSE_MAIN_RADIUS, PULSE_MAIN_RADIUS, PULSE_CYLINDER_HEIGHT, 16);
        const pulseMat = new THREE.MeshBasicMaterial({ color: color });
        const pulseMesh = new THREE.Mesh(pulseGeom, pulseMat);
        const glowGeom = new THREE.CylinderGeometry(PULSE_GLOW_RADIUS, PULSE_GLOW_RADIUS, PULSE_CYLINDER_HEIGHT, 16);
        const glowMat = new THREE.MeshBasicMaterial({ color: 0x80DEEA, transparent: true, opacity: 0.3 });
        pulseMesh.add(new THREE.Mesh(glowGeom, glowMat));
        const pulse = { mesh: pulseMesh, curve: curve, progress: 0, speed: 0.008 };
        activePulses.push(pulse);
        earthMesh.add(pulseMesh);
    }
    
    // --- FIX 2: Make pulse triggers robust against race conditions ---
    function triggerPulse(clientId, retries = 5) {
        if (retries <= 0) return; // Stop if we can't find the arc
        const key = `arc-${clientId}`;
        if (activeArcs.has(key)) {
            createPulse(activeArcs.get(key).curve, 0x0077FF);
        } else {
            // Arc doesn't exist yet, wait and try again
            setTimeout(() => triggerPulse(clientId, retries - 1), 100);
        }
    }
    
    function triggerBroadcastPulse(clientId, retries = 5) {
        if (retries <= 0) return;
        const key = `arc-${clientId}`;
        if (activeArcs.has(key)) {
            const originalCurve = activeArcs.get(key).curve;
            const broadcastCurve = new THREE.CubicBezierCurve3(originalCurve.v3, originalCurve.v2, originalCurve.v1, originalCurve.v0);
            createPulse(broadcastCurve, 0xFFD700);
        } else {
            setTimeout(() => triggerBroadcastPulse(clientId, retries - 1), 100);
        }
    }

    function animatePulses() { /* ... (no changes needed) ... */
        for (let i = activePulses.length - 1; i >= 0; i--) {
            const pulse = activePulses[i];
            pulse.progress += pulse.speed;
            if (pulse.progress >= 1) {
                earthMesh.remove(pulse.mesh);
                pulse.mesh.traverse(child => { if (child.geometry) child.geometry.dispose(); if (child.material) child.material.dispose(); });
                activePulses.splice(i, 1);
            } else {
                pulse.mesh.position.copy(pulse.curve.getPoint(pulse.progress));
                const tangent = pulse.curve.getTangent(pulse.progress).normalize();
                if (!tangent.equals(new THREE.Vector3(0,0,0))) {
                    const quaternion = new THREE.Quaternion();
                    quaternion.setFromUnitVectors(cylinderUp, tangent);
                    pulse.mesh.quaternion.copy(quaternion);
                }
            }
        }
    }

    function triggerServerGlow() { /* ... (no changes needed) ... */
        if (!serverGlow) return;
        new TWEEN.Tween(serverGlow.material).to({ opacity: 0.7 }, 300).easing(TWEEN.Easing.Quadratic.Out).yoyo(true).repeat(1).delay(100).start();
    }
    
    function animate() {
        // --- FIX 1: Capture the frame ID ---
        animationFrameId = requestAnimationFrame(animate);
        TWEEN.update();
        if (cloudsMesh && cloudsMesh.visible) cloudsMesh.rotation.y += 0.0001;
        animatePulses();
        controls.update();
        composer.render();
    }

    // ... (onWindowResize, onMouseMove, updateClientPoints, etc. remain the same)
    function onWindowResize() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        composer.setSize(container.clientWidth, container.clientHeight);
    }

    function onMouseMove(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const pointsToCheck = [...clientPoints.values(), serverPoint].filter(p => p && p.visible);
        const intersects = raycaster.intersectObjects(pointsToCheck);

        if (intersects.length > 0) {
            const intersected = intersects[0].object;
            if (currentlyHovered !== intersected) {
                currentlyHovered = intersected;
                const data = intersected.userData;
                let content = `<strong style="color: ${data.type === 'server' ? '#AB74FF' : '#2ECC71'};">${data.type === 'server' ? 'Server' : 'Client'} #${data.id}</strong>`;
                content += `<div>Location: ${data.location.name}</div>`;
                if (data.tokenomics) {
                    const balance = (data.tokenomics.balance || 0).toFixed(2);
                    const stake = (data.tokenomics.total_stake || 0).toFixed(2);
                    content += `<div style="margin-top: 5px;">Balance: ${balance} PHOENIX</div>`;
                    content += `<div>Stake: ${stake} PHOENIX</div>`;
                }
                tooltipElement.innerHTML = content;
                tooltipElement.style.display = 'block';
            }
            tooltipElement.style.left = `${event.clientX - rect.left + 15}px`;
            tooltipElement.style.top = `${event.clientY - rect.top + 15}px`;
        } else {
            if (currentlyHovered) {
                currentlyHovered = null;
                tooltipElement.style.display = 'none';
            }
        }
    }
    
    function updateClientPoints(connectedClients, tokenomicsData) {
        const connectedClientIds = connectedClients.map(c => c.id);
        connectedClients.forEach(client => {
            if (client.location) {
                const pos = latLonToVector3(client.location.lat, client.location.lon, GLOBE_RADIUS);
                let point = clientPoints.get(client.id);
                if (!point) {
                    const pointGeometry = new THREE.SphereGeometry(CLIENT_POINT_RADIUS, 16, 16);
                    const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x2ECC71 });
                    point = new THREE.Mesh(pointGeometry, pointMaterial);
                    point.position.copy(pos);
                    earthMesh.add(point);
                    clientPoints.set(client.id, point);
                }
                point.visible = true;
                point.userData = { type: 'client', id: client.id, location: client.location, tokenomics: tokenomicsData[client.id] };
            }
        });
        clientPoints.forEach((point, id) => { if (!connectedClientIds.includes(id)) point.visible = false; });
    }

    function updateServerPoint(serverLocation) {
        if (serverPoint) { earthMesh.remove(serverPoint); serverPoint.geometry.dispose(); serverPoint.material.dispose(); serverGlow = null; }
        if (serverLocation) {
            const pos = latLonToVector3(serverLocation.lat, serverLocation.lon, GLOBE_RADIUS);
            const geometry = new THREE.SphereGeometry(SERVER_POINT_RADIUS, 16, 16);
            const material = new THREE.MeshBasicMaterial({ color: 0x8A3FFC });
            serverPoint = new THREE.Mesh(geometry, material);
            serverPoint.position.copy(pos);
            serverPoint.userData = { type: 'server', id: serverLocation.name, location: serverLocation };
            earthMesh.add(serverPoint);

            const glowGeom = new THREE.SphereGeometry(SERVER_POINT_RADIUS * 2.5, 32, 32);
            const glowMat = new THREE.MeshBasicMaterial({ color: 0x8A3FFC, transparent: true, opacity: 0 });
            serverGlow = new THREE.Mesh(glowGeom, glowMat);
            serverPoint.add(serverGlow);
        }
    }

    function flyTo(targetLocation) {
        const targetPosition = latLonToVector3(targetLocation.lat, targetLocation.lon, 200);
        const start = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
        new TWEEN.Tween(start).to(targetPosition, 1500).easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => { camera.position.set(start.x, start.y, start.z); controls.target.set(0, 0, 0); }).start();
    }
    
    function toggleClouds(visible) { if (cloudsMesh) cloudsMesh.visible = visible; }
    function toggleRotation(enabled) { if (controls) controls.autoRotate = enabled; }
    
    // --- FIX 1: The crucial cleanup function ---
    function destroy() {
        console.log("Destroying Globe instance and cleaning up resources...");
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener('resize', onWindowResize);
        if (renderer) {
            renderer.domElement.removeEventListener('mousemove', onMouseMove);
            renderer.dispose();
             if (renderer.domElement.parentElement) {
                renderer.domElement.parentElement.removeChild(renderer.domElement);
            }
        }
        if (scene) {
            scene.traverse(object => {
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }
        if (tooltipElement) tooltipElement.remove();
        // Clear all internal state
        scene = null; camera = null; renderer = null; controls = null; composer = null;
        clientPoints.clear(); activeArcs.clear(); activePulses = [];
    }

    const tweenScript = document.createElement('script');
    tweenScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/tween.js/18.6.4/tween.umd.js';
    tweenScript.onload = init;
    document.head.appendChild(tweenScript);

    return {
        updateClientPoints, updateServerPoint, addOrUpdateArc, removeInactiveArcs,
        clearAllArcs, triggerPulse, flyTo, toggleClouds, toggleRotation,
        triggerServerGlow, triggerBroadcastPulse,
        destroy // --- Expose the destroy method ---
    };
}