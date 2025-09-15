const GLOBE_RADIUS = 100;

function createGlobe(container) {
    let scene, camera, renderer, controls, earthMesh, cloudsMesh, composer;
    const clientPoints = new Map();
    
    // Hardcoded client locations (lat, lon) for demonstration
    const clientLocations = [
        { lat: 34.0522, lon: -118.2437 }, { lat: 40.7128, lon: -74.0060 },
        { lat: 51.5074, lon: -0.1278 },   { lat: 48.8566, lon: 2.3522 },
        { lat: 35.6895, lon: 139.6917 },  { lat: -33.8688, lon: 151.2093 },
        { lat: 19.0760, lon: 72.8777 },   { lat: -23.5505, lon: -46.6333 },
        { lat: 55.7558, lon: 37.6173 },   { lat: 39.9042, lon: 116.4074 }
    ];

    function latLonToVector3(lat, lon, radius) {
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);
        const x = -(radius * Math.sin(phi) * Math.cos(theta));
        const z = (radius * Math.sin(phi) * Math.sin(theta));
        const y = (radius * Math.cos(phi));
        return new THREE.Vector3(x, y, z);
    }

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
            opacity: 0.0
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
        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x4A90E2, 0.5);
        scene.add(hemisphereLight);
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        // Add client points
        for (let i = 0; i < 10; i++) {
            const pos = latLonToVector3(clientLocations[i].lat, clientLocations[i].lon, GLOBE_RADIUS);
            const pointGeometry = new THREE.SphereGeometry(1.5, 16, 16);
            const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x8A93A2, transparent: true, opacity: 0.5 });
            const point = new THREE.Mesh(pointGeometry, pointMaterial);
            point.position.copy(pos);
            earthMesh.add(point);
            clientPoints.set(i.toString(), point);
        }

        // Post-processing
        const renderScene = new THREE.RenderPass(scene, camera);
        
        // --- FIX 2: TUNE BLOOM PARAMETERS FOR A SUBTLE TEAL GLOW ---
        const bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.0,    // strength: lower for a softer glow
            0.75,    // radius: larger for a more diffuse glow
            0.25    // threshold: high, so only the brightest parts (atmosphere rim, lights) bloom
        );
        // -------------------------------------------------------------
        
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearAlpha(0.0);
        composer.addPass(renderScene);
        composer.addPass(bloomPass);

        animate();
        window.addEventListener('resize', onWindowResize);
    }

    function animate() {
        requestAnimationFrame(animate);
        if (cloudsMesh) {
            cloudsMesh.rotation.y += 0.0000;
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

    function updateClientStatus(activeClientIds) {
        clientPoints.forEach((point, id) => {
            if (activeClientIds.includes(parseInt(id))) {
                point.material.color.set(0x2ECC71); // Green for active
                point.material.opacity = 1.0;
            } else {
                point.material.color.set(0x8A93A2); // Gray for inactive
                point.material.opacity = 0.5;
            }
        });
    }

    init();

    return {
        updateClientStatus
    };
}