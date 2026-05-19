import * as THREE from "./assets/vendor/three.webgpu.js";

(() => {
  const overlay = document.querySelector("[data-intro-screen]");
  if (!overlay) return;

  const canvas = overlay.querySelector("[data-intro-canvas]");
  const enterButton = overlay.querySelector("[data-enter-site]");
  const enterLabel = overlay.querySelector("[data-enter-label]");
  const roomPage = document.querySelector(".room-page");

  const {
    normalWorldGeometry,
    output,
    texture,
    vec3,
    vec4,
    normalize,
    positionWorld,
    cameraPosition,
    color,
    uniform,
    mix,
  } = THREE.TSL;

  const TARGET = { lat: 40.634, lon: -74.52 };
  const DEBUG_CITIES = [
    { label: "New York", lat: 40.7128, lon: -74.006, color: 0x86fff0 },
    { label: "Los Angeles", lat: 34.0522, lon: -118.2437, color: 0x86fff0 },
    { label: "San Francisco", lat: 37.7749, lon: -122.4194, color: 0x86fff0 },
    { label: "Seattle", lat: 47.6062, lon: -122.3321, color: 0x86fff0 },
    { label: "Chicago", lat: 41.8781, lon: -87.6298, color: 0x86fff0 },
    { label: "Boston", lat: 42.3601, lon: -71.0589, color: 0x86fff0 },
    { label: "北京", lat: 39.9042, lon: 116.4074, color: 0x86fff0 },
    { label: "上海", lat: 31.2304, lon: 121.4737, color: 0x86fff0 },
    { label: "深圳", lat: 22.5431, lon: 114.0579, color: 0x86fff0 },
    { label: "香港", lat: 22.3193, lon: 114.1694, color: 0x86fff0 },
    { label: "Singapore", lat: 1.3521, lon: 103.8198, color: 0x86fff0 },
    { label: "Toronto", lat: 43.6532, lon: -79.3832, color: 0x86fff0 },
    { label: "Vancouver", lat: 49.2827, lon: -123.1207, color: 0x86fff0 },
    { label: "London", lat: 51.5072, lon: -0.1276, color: 0x86fff0 },
    { label: "Berlin", lat: 52.52, lon: 13.405, color: 0x86fff0 },
    { label: "München", lat: 48.1351, lon: 11.582, color: 0x86fff0 },
    { label: "東京", lat: 35.6762, lon: 139.6503, color: 0x86fff0 },
    { label: "大阪", lat: 34.6937, lon: 135.5023, color: 0x86fff0 },
    { label: "서울", lat: 37.5665, lon: 126.978, color: 0x86fff0 },
    { label: "台北", lat: 25.033, lon: 121.5654, color: 0x86fff0 },
    { label: "मुंबई", lat: 19.076, lon: 72.8777, color: 0x86fff0 },
    { label: "Bengaluru", lat: 12.9716, lon: 77.5946, color: 0x86fff0 },
    { label: "Paris", lat: 48.8566, lon: 2.3522, color: 0x86fff0 },
    { label: "Москва", lat: 55.7558, lon: 37.6173, color: 0x86fff0 },
    { label: "Zürich", lat: 47.3769, lon: 8.5417, color: 0x86fff0 },
    { label: "Amsterdam", lat: 52.3676, lon: 4.9041, color: 0x86fff0 },
    { label: "Helsinki", lat: 60.1699, lon: 24.9384, color: 0x86fff0 },
    { label: "Madrid", lat: 40.4168, lon: -3.7038, color: 0x86fff0 },
    { label: "Sydney", lat: -33.8688, lon: 151.2093, color: 0x86fff0 },
    { label: "Melbourne", lat: -37.8136, lon: 144.9631, color: 0x86fff0 },
    { label: "Мінск", lat: 53.9006, lon: 27.559, color: 0x86fff0 },
    { label: "Hà Nội", lat: 21.0278, lon: 105.8342, color: 0x86fff0 },
  ];
  const state = {
    closed: false,
    visitor: { lat: 39.5, lon: -98.35 },
    dragging: false,
    lastPointer: { x: 0, y: 0 },
    velocity: { x: 0, y: 0 },
  };

  const clock = new THREE.Clock();
  const START_TIME = performance.now();

  document.body.classList.add("intro-active");
  document.body.dataset.introRenderer = "webgpu-tsl";
  roomPage?.setAttribute("aria-hidden", "true");

  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(25, 1, 0.1, 100);
  camera.position.set(3.9, 1.62, 2.58);
  camera.lookAt(0, 0, 0);

  let sunAnchor = calculateSubsolarVector(new Date());
  let lastSunRefresh = performance.now();
  const sun = new THREE.DirectionalLight("#ffffff", 2);
  scene.add(sun);

  const atmosphereDayColor = uniform(color("#4db2ff"));
  const atmosphereTwilightColor = uniform(color("#bc490b"));
  const sunDirection = uniform(new THREE.Vector3(0, 0, 1));

  const textureLoader = new THREE.TextureLoader();
  const textureBase = "./assets/earth";

  const dayTexture = textureLoader.load(`${textureBase}/earth_day_4096.jpg`);
  dayTexture.colorSpace = THREE.SRGBColorSpace;
  dayTexture.anisotropy = 8;

  const nightTexture = textureLoader.load(`${textureBase}/earth_night_4096.jpg`);
  nightTexture.colorSpace = THREE.SRGBColorSpace;
  nightTexture.anisotropy = 8;

  const viewDirection = positionWorld.sub(cameraPosition).normalize();
  const fresnel = viewDirection.dot(normalWorldGeometry).abs().oneMinus().toVar();
  const sunOrientation = normalWorldGeometry.dot(normalize(sunDirection)).toVar();
  const atmosphereColor = mix(atmosphereTwilightColor, atmosphereDayColor, sunOrientation.smoothstep(-0.25, 0.75));

  const sphereGeometry = new THREE.SphereGeometry(1, 64, 64);
  const globeMaterial = new THREE.MeshStandardNodeMaterial();
  const day = texture(dayTexture).rgb;
  globeMaterial.colorNode = day;

  const night = texture(nightTexture);
  const dayStrength = sunOrientation.smoothstep(-0.25, 0.5);
  const atmosphereDayStrength = sunOrientation.smoothstep(-0.5, 1);
  const atmosphereMix = atmosphereDayStrength.mul(fresnel.pow(2)).clamp(0, 1);
  let finalOutput = mix(night.rgb, day, dayStrength);
  finalOutput = mix(finalOutput, atmosphereColor, atmosphereMix);
  globeMaterial.outputNode = vec4(finalOutput, output.a);

  const globe = new THREE.Mesh(sphereGeometry, globeMaterial);
  globe.rotation.y = getGlobeFocusRotation(TARGET);
  scene.add(globe);
  updateSunPosition();

  const atmosphereMaterial = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, transparent: true });
  let alpha = fresnel.remap(0.73, 1, 1, 0).pow(3);
  alpha = alpha.mul(sunOrientation.smoothstep(-0.5, 1));
  atmosphereMaterial.outputNode = vec4(atmosphereColor, alpha);

  const atmosphere = new THREE.Mesh(sphereGeometry, atmosphereMaterial);
  atmosphere.scale.setScalar(1.04);
  scene.add(atmosphere);

  const routeGroup = new THREE.Group();
  globe.add(routeGroup);
  const routeSegments = [];
  const routeCurve = [];
  let routeBackdrop = null;

  const routeMaterialOuter = new THREE.MeshBasicMaterial({
    color: 0x65fff0,
    transparent: true,
    opacity: 0.06,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const routeMaterialInner = new THREE.MeshBasicMaterial({
    color: 0xf8fffe,
    transparent: true,
    opacity: 0.22,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const sourcePin = makeEndpointMarker(0x65fff0);
  const targetPin = makeEndpointMarker(0xf0b35b);
  globe.add(sourcePin, targetPin);

  const cityMarkerGroup = new THREE.Group();
  DEBUG_CITIES.forEach((city) => {
    const marker = makeCityMarker(city.label, city.color);
    marker.userData.surfaceNormal = geoToVector(city.lat, city.lon, 1).normalize();
    setPinPosition(marker, city.lat, city.lon, 1.025);
    cityMarkerGroup.add(marker);
  });
  globe.add(cityMarkerGroup);
  updateCityMarkerVisibility();

  const pulse = new THREE.Mesh(
    new THREE.SphereGeometry(0.015, 18, 18),
    new THREE.MeshBasicMaterial({
      color: 0xf9fffe,
      transparent: true,
      opacity: 0.82,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  globe.add(pulse);

  const stars = createStars(300);
  scene.add(stars);

  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  resize();

  buildRoute(state.visitor, TARGET);
  updateEnterLabel(state.visitor);
  enterButton.addEventListener("click", closeIntro);
  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerUp);
  canvas.addEventListener("pointercancel", handlePointerUp);
  canvas.addEventListener("lostpointercapture", handlePointerUp);
  window.addEventListener("resize", resize);
  window.addEventListener("orientationchange", resize);

  Promise.race([
    fetchVisitorLocation(),
    new Promise((resolve) => setTimeout(() => resolve(null), 2600)),
  ]).then((location) => {
    if (state.closed || !location) return;
    state.visitor = location;
    buildRoute(location, TARGET);
    updateEnterLabel(location);
  }).catch(() => {});

  renderer.setAnimationLoop(animate);

  async function fetchVisitorLocation() {
    if (isLocalPreview()) {
      return { lat: -33.8688, lon: 151.2093 };
    }

    const providers = [
      async () => {
        const response = await fetch("https://ipapi.co/json/", { cache: "no-store" });
        if (!response.ok) throw new Error("ipapi failed");
        const data = await response.json();
        if (!data?.latitude || !data?.longitude) throw new Error("ipapi missing coordinates");
        return { lat: Number(data.latitude), lon: Number(data.longitude) };
      },
      async () => {
        const response = await fetch("https://ipwho.is/?lang=en", { cache: "no-store" });
        if (!response.ok) throw new Error("ipwho failed");
        const data = await response.json();
        if (!data?.success) throw new Error("ipwho failed");
        return { lat: Number(data.latitude), lon: Number(data.longitude) };
      },
    ];

    for (const provider of providers) {
      try {
        const location = await provider();
        if (Number.isFinite(location.lat) && Number.isFinite(location.lon)) return location;
      } catch {
        // Try the next provider.
      }
    }
    return null;
  }

  function isLocalPreview() {
    return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  }

  function updateEnterLabel(location) {
    if (!enterLabel) return;
    const miles = Math.round(distanceMiles(location, TARGET)).toLocaleString("en-US");
    enterLabel.textContent = `Welcome to visit Yu Guo from ${miles} miles away`;
  }

  function distanceMiles(start, end) {
    const earthRadiusMiles = 3958.8;
    const lat1 = THREE.MathUtils.degToRad(start.lat);
    const lat2 = THREE.MathUtils.degToRad(end.lat);
    const dLat = lat2 - lat1;
    const dLon = THREE.MathUtils.degToRad(end.lon - start.lon);
    const a = Math.sin(dLat / 2) ** 2
      + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
    return earthRadiusMiles * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function resize() {
    const rect = overlay.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }

  function createStars(count) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const radius = 18 + Math.random() * 45;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
      positions[i * 3 + 0] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        color: 0xeafcff,
        size: 0.02,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      }),
    );
  }

  function geoToVector(lat, lon, radius = 1) {
    const latRad = lat * Math.PI / 180;
    const lonRad = lon * Math.PI / 180;
    return new THREE.Vector3(
      radius * Math.cos(latRad) * Math.cos(lonRad),
      radius * Math.sin(latRad),
      -radius * Math.cos(latRad) * Math.sin(lonRad),
    );
  }

  function calculateSubsolarVector(date) {
    const start = Date.UTC(date.getUTCFullYear(), 0, 0);
    const dayOfYear = Math.floor((date.getTime() - start) / 86400000);
    const utcHours = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
    const gamma = (2 * Math.PI / 365) * (dayOfYear - 1 + (utcHours - 12) / 24);
    const equationOfTime = 229.18 * (
      0.000075 +
      0.001868 * Math.cos(gamma) -
      0.032077 * Math.sin(gamma) -
      0.014615 * Math.cos(2 * gamma) -
      0.040849 * Math.sin(2 * gamma)
    );
    const declination = 180 / Math.PI * (
      0.006918 -
      0.399912 * Math.cos(gamma) +
      0.070257 * Math.sin(gamma) -
      0.006758 * Math.cos(2 * gamma) +
      0.000907 * Math.sin(2 * gamma) -
      0.002697 * Math.cos(3 * gamma) +
      0.00148 * Math.sin(3 * gamma)
    );
    const subsolarLongitude = THREE.MathUtils.euclideanModulo(720 - (utcHours * 60 + equationOfTime) / 4, 360) - 180;
    return geoToVector(declination, subsolarLongitude, 1);
  }

  function updateSunPosition() {
    const sunVector = sunAnchor.clone().applyQuaternion(globe.quaternion).normalize();
    sunDirection.value.copy(sunVector);
    sun.position.copy(sunVector.multiplyScalar(3));
    sun.target.position.set(0, 0, 0);
    sun.target.updateMatrixWorld();
  }

  function buildRoute(start, end) {
    routeSegments.forEach((segment) => routeGroup.remove(segment.outer, segment.inner));
    routeSegments.length = 0;
    routeCurve.length = 0;
    if (routeBackdrop) {
      routeGroup.remove(routeBackdrop);
      routeBackdrop.geometry.dispose();
      routeBackdrop.material.dispose();
      routeBackdrop = null;
    }

    const routeRadius = 1.016;
    const count = 120;
    const lonDelta = shortestLongitudeDelta(start.lon, end.lon);
    for (let i = 0; i < count; i += 1) {
      const t = i / (count - 1);
      routeCurve.push(routePoint(start, end, lonDelta, t, routeRadius));
    }

    const backdropCurve = new THREE.CatmullRomCurve3(routeCurve.map((point) => point.clone()));
    routeBackdrop = new THREE.Mesh(
      new THREE.TubeGeometry(backdropCurve, 160, 0.006, 8, false),
      new THREE.MeshBasicMaterial({
        color: 0x67fff0,
        transparent: true,
        opacity: 0.12,
        depthTest: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    routeBackdrop.renderOrder = 2;
    routeGroup.add(routeBackdrop);

    const up = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < routeCurve.length - 1; i += 1) {
      const a = routeCurve[i].clone();
      const b = routeCurve[i + 1].clone();
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length();
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);

      const outer = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, len, 8, 1, true), routeMaterialOuter.clone());
      outer.position.copy(mid);
      outer.quaternion.setFromUnitVectors(up, dir.clone().normalize());
      outer.visible = true;
      outer.renderOrder = 3;

      const inner = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, len, 8, 1, true), routeMaterialInner.clone());
      inner.position.copy(mid);
      inner.quaternion.setFromUnitVectors(up, dir.clone().normalize());
      inner.visible = true;
      inner.renderOrder = 4;

      routeGroup.add(outer, inner);
      routeSegments.push({ outer, inner });
    }

    setPinPosition(sourcePin, start.lat, start.lon, 1.028);
    setPinPosition(targetPin, end.lat, end.lon, 1.028);
  }

  function shortestLongitudeDelta(startLon, endLon) {
    return THREE.MathUtils.euclideanModulo(endLon - startLon + 540, 360) - 180;
  }

  function routePoint(start, end, lonDelta, t, radius) {
    const lat = THREE.MathUtils.lerp(start.lat, end.lat, t);
    const lon = start.lon + lonDelta * t;
    const lift = Math.sin(t * Math.PI) * 0.012;
    return geoToVector(lat, lon, radius + lift);
  }

  function makeEndpointMarker(markerColor) {
    const group = new THREE.Group();
    const texture = createEndpointTexture(markerColor);
    const icon = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: texture,
        color: 0xffffff,
        transparent: true,
        opacity: 0.98,
        depthTest: true,
        depthWrite: false,
      }),
    );
    icon.center.set(0.5, 0.5);
    icon.scale.set(0.072, 0.072, 1);
    icon.renderOrder = 12;
    group.add(icon);
    return group;
  }

  function makeCityMarker(label, markerColor) {
    const group = new THREE.Group();
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.008, 10, 10),
      new THREE.MeshBasicMaterial({
        color: markerColor,
        transparent: true,
        opacity: 0.96,
        depthTest: true,
        depthWrite: false,
      }),
    );
    const labelSprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: createTextTexture(label, markerColor),
        transparent: true,
        depthTest: true,
        depthWrite: false,
      }),
    );
    labelSprite.position.set(0.03, 0.025, 0);
    labelSprite.scale.set(0.19, 0.064, 1);
    group.add(dot, labelSprite);
    return group;
  }

  function createTextTexture(text, markerColor) {
    const width = 320;
    const height = 96;
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = width;
    labelCanvas.height = height;
    const ctx = labelCanvas.getContext("2d");
    ctx.font = "700 46px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0, 0, 0, 0.95)";
    ctx.shadowBlur = 10;
    ctx.lineWidth = 5;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.86)";
    ctx.strokeText(text, 28, height / 2 + 1);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, 28, height / 2 + 1);

    const texture = new THREE.CanvasTexture(labelCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  function createEndpointTexture(markerColor) {
    const size = 256;
    const pinCanvas = document.createElement("canvas");
    pinCanvas.width = size;
    pinCanvas.height = size;
    const ctx = pinCanvas.getContext("2d");
    const colorValue = `#${markerColor.toString(16).padStart(6, "0")}`;

    ctx.save();
    ctx.translate(128, 128);
    ctx.shadowColor = colorValue;
    ctx.shadowBlur = 26;
    ctx.strokeStyle = colorValue;
    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.lineWidth = 9;
    ctx.beginPath();
    ctx.moveTo(0, -70);
    ctx.lineTo(70, 0);
    ctx.lineTo(0, 70);
    ctx.lineTo(-70, 0);
    ctx.closePath();
    ctx.stroke();

    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.78)";
    ctx.stroke();
    ctx.shadowBlur = 10;
    ctx.fillStyle = colorValue;
    ctx.beginPath();
    ctx.arc(0, 0, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.beginPath();
    ctx.arc(0, 0, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    const texture = new THREE.CanvasTexture(pinCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  function setPinPosition(group, lat, lon, radius) {
    const point = geoToVector(lat, lon, radius);
    group.position.copy(point);
    group.lookAt(point.clone().multiplyScalar(1.2));
  }

  function updateCityMarkerVisibility() {
    const cameraDirection = camera.position.clone().normalize();
    cityMarkerGroup.children.forEach((marker) => {
      const normal = marker.userData.surfaceNormal
        ?.clone()
        .applyQuaternion(globe.quaternion)
        .normalize();
      const visibility = normal ? normal.dot(cameraDirection) : 1;
      const opacity = THREE.MathUtils.smoothstep(visibility, 0.5, 0.62);
      marker.visible = opacity > 0.02;
      marker.children.forEach((child) => {
        if (child.material) child.material.opacity = opacity;
      });
    });
  }

  function getGlobeFocusRotation(location) {
    const point = geoToVector(location.lat, location.lon, 1);
    const pointAngle = Math.atan2(point.x, point.z);
    const cameraAngle = Math.atan2(camera.position.x, camera.position.z);
    return cameraAngle - pointAngle;
  }

  function handlePointerDown(event) {
    if (state.closed) return;
    state.dragging = true;
    state.lastPointer.x = event.clientX;
    state.lastPointer.y = event.clientY;
    state.velocity.x = 0;
    state.velocity.y = 0;
    canvas.setPointerCapture?.(event.pointerId);
  }

  function handlePointerMove(event) {
    if (!state.dragging || state.closed) return;
    const dx = event.clientX - state.lastPointer.x;
    const dy = event.clientY - state.lastPointer.y;
    state.lastPointer.x = event.clientX;
    state.lastPointer.y = event.clientY;
    const rotateSpeed = 0.006;
    globe.rotation.y += dx * rotateSpeed;
    globe.rotation.x = THREE.MathUtils.clamp(globe.rotation.x + dy * rotateSpeed, -0.82, 0.82);
    state.velocity.x = dx * rotateSpeed;
    state.velocity.y = dy * rotateSpeed;
    updateSunPosition();
    updateCityMarkerVisibility();
  }

  function handlePointerUp(event) {
    state.dragging = false;
    canvas.releasePointerCapture?.(event.pointerId);
  }

  function animate() {
    if (state.closed) return;
    const delta = clock.getDelta();
    const elapsed = performance.now() - START_TIME;
    if (!state.dragging) {
      globe.rotation.y += state.velocity.x;
      globe.rotation.x = THREE.MathUtils.clamp(globe.rotation.x + state.velocity.y, -0.82, 0.82);
      state.velocity.x *= 0.92;
      state.velocity.y *= 0.92;
    }
    stars.rotation.y += delta * 0.006;
    if (performance.now() - lastSunRefresh > 60000) {
      sunAnchor = calculateSubsolarVector(new Date());
      lastSunRefresh = performance.now();
    }
    updateSunPosition();

    if (routeCurve.length > 1) {
      const t = (elapsed % 3200) / 3200;
      const index = Math.min(routeCurve.length - 1, Math.floor(t * (routeCurve.length - 1)));
      routeSegments.forEach((segment, segmentIndex) => {
        const trailDistance = index - segmentIndex;
        const trail = trailDistance >= 0 && trailDistance < 14 ? 1 - trailDistance / 14 : 0;
        segment.outer.material.opacity = 0.04 + trail * 0.1;
        segment.inner.material.opacity = 0.14 + trail * 0.42;
      });
      pulse.position.copy(routeCurve[index]);
      pulse.scale.setScalar(0.72 + 0.12 * Math.sin(elapsed / 140));
      pulse.material.opacity = 0.78;
    }

    renderer.render(scene, camera);
  }

  function closeIntro() {
    if (state.closed) return;
    state.closed = true;
    renderer.setAnimationLoop(null);
    overlay.classList.add("is-hidden");
    document.body.classList.remove("intro-active");
    roomPage?.removeAttribute("aria-hidden");
    window.setTimeout(() => overlay.remove(), 560);
  }
})();
