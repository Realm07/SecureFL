
const GLOBE_RADIUS = 100;

function latLonToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const z = (radius * Math.sin(phi) * Math.sin(theta));
    const y = (radius * Math.cos(phi));
    return new THREE.Vector3(x, y, z);
}

function createCurve(startVec, endVec) {
    const start = startVec;
    const end = endVec;
    const mid = start.clone().lerp(end, 0.5);
    const distance = start.distanceTo(end);
    mid.normalize().multiplyScalar(GLOBE_RADIUS + distance * 0.3);
    const curve = new THREE.CubicBezierCurve3(start, start.clone().normalize().multiplyScalar(GLOBE_RADIUS + 10), mid, end);
    return curve;
}

function createGlobe(container) {
    let scene, camera, renderer, controls, earthMesh, cloudsMesh, composer;
    let clientPoints = new Map(); // Stores { id: THREE.Mesh }
    let serverPoint = null;
    let activeArcs = new Map();
    let activePulses = [];

    function init() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
        camera.position.z = 250;

        // --- FIX 1: MAKE RENDERER TRANSPARENT ---
        // We still need alpha:true, but we also set the clearAlpha to 0.
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearAlpha(0.0); // This makes the renderer background transparent
        // ------------------------------------------
        
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        // Remove placeholder text
        const title = document.querySelector('.globe-title');
        const subtitle = document.querySelector('.globe-subtitle');
        if(title) title.style.display = 'none';
        if(subtitle) subtitle.style.display = 'none';

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.2;
        controls.enablePan = false;
        controls.minDistance = 200;
        controls.maxDistance = 500;

        const textureLoader = new THREE.TextureLoader();
        
        // 1. Earth's main texture (Night Map)
        const earthTexture = textureLoader.load('/static/textures/Earth_Night_Map_HIGH.jpg'); 
        
        // 2. Specular Map (for shininess)
        const specularMap = textureLoader.load('/static/textures/Earth_Specular_Map_HIGH.tif'); 
        const earthMaterial = new THREE.MeshPhongMaterial({
            map: earthTexture,
            specularMap: specularMap,
            specular: new THREE.Color('#111111'),
            shininess: 5,
        });
        const earthGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
        earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
        scene.add(earthMesh);


        // 3. Clouds Layer
        const cloudTexture = textureLoader.load('/static/textures/Earth_Clouds_HIGH.jpg');
        const cloudMaterial = new THREE.MeshLambertMaterial({
            map: cloudTexture,
            transparent: true,
            opacity: 0.2
        });
        const cloudGeometry = new THREE.SphereGeometry(GLOBE_RADIUS + 2, 64, 64);
        cloudsMesh = new THREE.Mesh(cloudGeometry, cloudMaterial);
        scene.add(cloudsMesh);

        // Atmosphere Glow (using shader for rim effect)
        const atmosphereMaterial = new THREE.ShaderMaterial({
            vertexShader: `
                varying vec3 vNormal;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                varying vec3 vNormal;
                void main() {
                    float intensity = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
                    gl_FragColor = vec4(0.0, 0.70, 0.85, 1.0) * intensity;
                }
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true
        });
        const atmosphereGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.04, 64, 64);
        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        scene.add(atmosphere);

        // Lighting
        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x4A90E2, 0.6);
        scene.add(hemisphereLight);
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        scene.add(ambientLight);

        // --- FIX: REMOVED THE BUGGY for-LOOP THAT USED clientLocations ---
        // Client points are now created dynamically in updateClientAndServerPoints.

        // --- FIX: CORRECT POST-PROCESSING INITIALIZATION ORDER ---
        // 1. Create the EffectComposer FIRST, passing the renderer to it.
        composer = new THREE.EffectComposer(renderer);
        
        // 2. Create the passes.
        const renderScene = new THREE.RenderPass(scene, camera);
        const bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            0.4, 0.5, 0.85
        );

        // 3. Add the passes to the composer.
        composer.addPass(renderScene);
        composer.addPass(bloomPass);
        // --------------------------------------------------------

        animate();
        window.addEventListener('resize', onWindowResize);
    }

    function addOrUpdateArc(clientId, clientLocation, serverLocation) {
        const key = `arc-${clientId}`;
        if (activeArcs.has(key)) {
            // Arc already exists, maybe update it later if needed
            return;
        }

        const startVec = latLonToVector3(clientLocation.lat, clientLocation.lon, GLOBE_RADIUS);
        const endVec = latLonToVector3(serverLocation.lat, serverLocation.lon, GLOBE_RADIUS);
        
        const curve = createCurve(startVec, endVec);
        const points = curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x00B4D8, transparent: true, opacity: 0.5 });
        
        const arc = new THREE.Line(geometry, material);
        earthMesh.add(arc);
        activeArcs.set(key, { arc, curve });
        
        // Create a pulse for this new arc
        createPulse(curve);
    }

    function removeArc(clientId) {
        const key = `arc-${clientId}`;
        if (activeArcs.has(key)) {
            const { arc } = activeArcs.get(key);
            earthMesh.remove(arc);
            arc.geometry.dispose();
            arc.material.dispose();
            activeArcs.delete(key);
        }
    }

    function clearAllArcs() {
        activeArcs.forEach((_, key) => {
            const clientId = key.split('-')[1];
            removeArc(clientId);
        });
        // Also clear any lingering pulses
        activePulses.forEach(pulse => earthMesh.remove(pulse.mesh));
        activePulses = [];
    }

    // --- NEW: PulseManager ---
    function createPulse(curve) {
        const geometry = new THREE.SphereGeometry(2, 8, 8);
        const material = new THREE.MeshBasicMaterial({ color: 0x2ECC71 });
        const pulseMesh = new THREE.Mesh(geometry, material);
        
        const pulse = {
            mesh: pulseMesh,
            curve: curve,
            progress: 0,
            speed: 0.005 + Math.random() * 0.005 // Randomize speed
        };
        activePulses.push(pulse);
        earthMesh.add(pulseMesh);
    }
    
    function animatePulses() {
        for (let i = activePulses.length - 1; i >= 0; i--) {
            const pulse = activePulses[i];
            pulse.progress += pulse.speed;

            if (pulse.progress >= 1) {
                // Reset the pulse to the beginning
                pulse.progress = 0;
            }
            
            const newPos = pulse.curve.getPoint(pulse.progress);
            pulse.mesh.position.copy(newPos);
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        if (cloudsMesh) {
            cloudsMesh.rotation.y += 0.0001;
        }
        animatePulses();
        if (serverPoint) {
            const time = Date.now() * 0.005;
            const scale = 1.0 + Math.sin(time) * 0.1; // Pulsate between 0.9 and 1.1 scale
            serverPoint.scale.set(scale, scale, scale);
        }
        controls.update();
        composer.render();
    }

    function onWindowResize() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
    }

    function updateClientAndServerPoints(connectedClients, serverLocation) {
        const connectedClientIds = connectedClients.map(c => c.id);

        // 1. Add or update points for currently connected clients
        connectedClients.forEach(client => {
            if (client.location) {
                const pos = latLonToVector3(client.location.lat, client.location.lon, GLOBE_RADIUS);
                
                if (clientPoints.has(client.id)) {
                    // Point already exists, just make sure it's visible
                    clientPoints.get(client.id).visible = true;
                } else {
                    // Point doesn't exist, create it
                    const pointGeometry = new THREE.SphereGeometry(1.5, 16, 16);
                    const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x2ECC71, transparent: true, opacity: 1.0 });
                    const point = new THREE.Mesh(pointGeometry, pointMaterial);
                    point.position.copy(pos);
                    earthMesh.add(point);
                    clientPoints.set(client.id, point);
                }
            }
        });
        
        // 2. Hide points for clients that have disconnected
        clientPoints.forEach((point, id) => {
            if (!connectedClientIds.includes(id)) {
                point.visible = false;
            }
        });

        // 3. Update server point
        if (serverPoint) {
            earthMesh.remove(serverPoint);
            serverPoint.geometry.dispose();
            serverPoint.material.dispose();
            serverPoint = null;
        }
        if (serverLocation) {
            const pos = latLonToVector3(serverLocation.lat, serverLocation.lon, GLOBE_RADIUS);
            
            // --- THE FIX: CHANGE CUBE TO SPHERE ---
            const geometry = new THREE.SphereGeometry(3, 16, 16); // A larger sphere
            const material = new THREE.MeshBasicMaterial({ color: 0x8A3FFC });
            // ------------------------------------

            serverPoint = new THREE.Mesh(geometry, material);
            serverPoint.position.copy(pos);
            earthMesh.add(serverPoint);
        }
    }

    init();

    return {
        updateClientAndServerPoints,
        addOrUpdateArc,
        removeArc,
        clearAllArcs
    };
}