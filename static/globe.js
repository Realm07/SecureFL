const GLOBE_RADIUS = 100;
const CLIENT_POINT_RADIUS = 0.7;
const SERVER_POINT_RADIUS = 2.0;
const PULSE_CYLINDER_HEIGHT = 5;
const PULSE_MAIN_RADIUS = 0.8;
// --- FIX: Increased glow radius for more effect ---
const PULSE_GLOW_RADIUS = 1.5; 
const ARC_THICKNESS = 0.25;


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

    const curve = new THREE.CubicBezierCurve3(startVec, controlPoint1, controlPoint2, endVec);
    return curve;
}


function createGlobe(container) {
    let scene, camera, renderer, controls, earthMesh, cloudsMesh, composer;
    let clientPoints = new Map();
    let serverPoint = null;
    let activeArcs = new Map();
    let activePulses = [];

    // --- FIX: Define the cylinder's default 'up' vector once ---
    const cylinderUp = new THREE.Vector3(0, 1, 0);

    function init() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
        camera.position.z = 250;

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearAlpha(0.0);
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
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
        const earthTexture = textureLoader.load('/static/textures/Earth_Night_Map_HIGH.jpg'); 
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

        const cloudTexture = textureLoader.load('/static/textures/Earth_Clouds_HIGH.jpg');
        const cloudMaterial = new THREE.MeshLambertMaterial({
            map: cloudTexture,
            transparent: true,
            opacity: 0.15
        });
        const cloudGeometry = new THREE.SphereGeometry(GLOBE_RADIUS + 0.5, 64, 64);
        cloudsMesh = new THREE.Mesh(cloudGeometry, cloudMaterial);
        scene.add(cloudsMesh);

        const atmosphereMaterial = new THREE.ShaderMaterial({
            vertexShader: `varying vec3 vNormal; void main() { vNormal = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
            fragmentShader: `varying vec3 vNormal; void main() { float intensity = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0); gl_FragColor = vec4(0.0, 0.70, 0.85, 1.0) * intensity; }`,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true
        });
        const atmosphereGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.01, 64, 64);
        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        scene.add(atmosphere);

        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x4A90E2, 0.6);
        scene.add(hemisphereLight);
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        scene.add(ambientLight);

        composer = new THREE.EffectComposer(renderer);
        const renderScene = new THREE.RenderPass(scene, camera);
        const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.7, 0.5, 0.85);
        composer.addPass(renderScene);
        composer.addPass(bloomPass);

        animate();
        window.addEventListener('resize', onWindowResize);
    }

    function addOrUpdateArc(clientId, clientLocation, serverLocation) {
        const key = `arc-${clientId}`;
        if (activeArcs.has(key)) return;

        const startVec = latLonToVector3(clientLocation.lat, clientLocation.lon, GLOBE_RADIUS);
        const endVec = latLonToVector3(serverLocation.lat, serverLocation.lon, GLOBE_RADIUS);
        const curve = createCurve(startVec, endVec);
        
        const geometry = new THREE.TubeGeometry(curve, 64, ARC_THICKNESS, 8, false);
        const material = new THREE.MeshBasicMaterial({ color: 0x00B4D8 });
        
        const arcMesh = new THREE.Mesh(geometry, material);
        earthMesh.add(arcMesh);
        activeArcs.set(key, { mesh: arcMesh, curve });
    }

    function removeInactiveArcs(connectedClientIds) {
        activeArcs.forEach((arcData, key) => {
            const arcClientId = parseInt(key.split('-')[1]);
            if (!connectedClientIds.includes(arcClientId)) {
                earthMesh.remove(arcData.mesh);
                arcData.mesh.geometry.dispose();
                arcData.mesh.material.dispose();
                activeArcs.delete(key);
            }
        });
    }

    function clearAllArcs() {
        activeArcs.forEach((arcData) => {
            earthMesh.remove(arcData.mesh);
            arcData.mesh.geometry.dispose();
            arcData.mesh.material.dispose();
        });
        activeArcs.clear();
    }

    function triggerPulse(clientId) {
        const key = `arc-${clientId}`;
        if (!activeArcs.has(key)) return;

        const { curve } = activeArcs.get(key);
        
        const pulseGeom = new THREE.CylinderGeometry(PULSE_MAIN_RADIUS, PULSE_MAIN_RADIUS, PULSE_CYLINDER_HEIGHT, 16);
        const pulseMat = new THREE.MeshBasicMaterial({ color: 0x0077FF });
        const pulseMesh = new THREE.Mesh(pulseGeom, pulseMat);

        const glowGeom = new THREE.CylinderGeometry(PULSE_GLOW_RADIUS, PULSE_GLOW_RADIUS, PULSE_CYLINDER_HEIGHT, 16);
        // --- FIX: Brighter color and lower opacity for better glow ---
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0x80DEEA, // Very light cyan
            transparent: true,
            opacity: 0.3
        });
        const glowMesh = new THREE.Mesh(glowGeom, glowMat);
        pulseMesh.add(glowMesh);
        
        const pulse = {
            mesh: pulseMesh,
            curve: curve,
            progress: 0,
            speed: 0.008,
            quaternion: new THREE.Quaternion() // Pre-create quaternion for performance
        };
        activePulses.push(pulse);
        earthMesh.add(pulseMesh);
    }
    
    function animatePulses() {
        for (let i = activePulses.length - 1; i >= 0; i--) {
            const pulse = activePulses[i];
            pulse.progress += pulse.speed;

            if (pulse.progress >= 1) {
                earthMesh.remove(pulse.mesh);
                pulse.mesh.traverse(child => {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) child.material.dispose();
                });
                activePulses.splice(i, 1);
            } else {
                const currentPos = pulse.curve.getPoint(pulse.progress);
                pulse.mesh.position.copy(currentPos);
                
                // --- FIX: Correctly orient the cylinder along the curve's tangent ---
                const tangent = pulse.curve.getTangent(pulse.progress).normalize();
                pulse.quaternion.setFromUnitVectors(cylinderUp, tangent);
                pulse.mesh.quaternion.copy(pulse.quaternion);
            }
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        if (cloudsMesh) cloudsMesh.rotation.y += 0.0001;
        
        animatePulses();

        if (serverPoint) {
            const time = Date.now() * 0.005;
            const scale = 1.0 + Math.sin(time) * 0.1;
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
        composer.setSize(container.clientWidth, container.clientHeight);
    }

    function updateClientPoints(connectedClients) {
        const connectedClientIds = connectedClients.map(c => c.id);
        connectedClients.forEach(client => {
            if (client.location) {
                const pos = latLonToVector3(client.location.lat, client.location.lon, GLOBE_RADIUS);
                if (clientPoints.has(client.id)) {
                    clientPoints.get(client.id).visible = true;
                } else {
                    const pointGeometry = new THREE.SphereGeometry(CLIENT_POINT_RADIUS, 16, 16);
                    const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x2ECC71 });
                    const point = new THREE.Mesh(pointGeometry, pointMaterial);
                    point.position.copy(pos);
                    earthMesh.add(point);
                    clientPoints.set(client.id, point);
                }
            }
        });
        clientPoints.forEach((point, id) => {
            if (!connectedClientIds.includes(id)) point.visible = false;
        });
    }
    
    function updateServerPoint(serverLocation) {
        if (serverPoint) {
            earthMesh.remove(serverPoint);
            serverPoint.geometry.dispose();
            serverPoint.material.dispose();
            serverPoint = null;
        }
        if (serverLocation) {
            const pos = latLonToVector3(serverLocation.lat, serverLocation.lon, GLOBE_RADIUS);
            const geometry = new THREE.SphereGeometry(SERVER_POINT_RADIUS, 16, 16);
            const material = new THREE.MeshBasicMaterial({ color: 0x8A3FFC });
            serverPoint = new THREE.Mesh(geometry, material);
            serverPoint.position.copy(pos);
            earthMesh.add(serverPoint);
        }
    }

    init();

    return {
        updateClientPoints,
        updateServerPoint,
        addOrUpdateArc,
        removeInactiveArcs,
        clearAllArcs,
        triggerPulse
    };
}