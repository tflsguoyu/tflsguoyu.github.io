(() => {
  const overlay = document.querySelector("[data-intro-screen]");
  if (!overlay) return;

  const canvas = overlay.querySelector("[data-intro-canvas]");
  const enterButton = overlay.querySelector("[data-enter-site]");
  const enterLabel = overlay.querySelector("[data-enter-label]");
  const roomPage = document.querySelector(".room-page");
  const ctx = canvas.getContext("2d", { alpha: true });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const TARGET = {
    lat: 40.634,
    lon: -74.52,
    label: "07059, US",
  };

  const state = {
    width: 0,
    height: 0,
    dpr: 1,
    closed: false,
    rotationOffset: 0,
    visitor: {
      lat: 39.5,
      lon: -98.35,
    },
    routePoints: [],
    earthImageReady: false,
  };

  const START_TIME = performance.now();
  const ROUTE_DRAW_MS = reducedMotion ? 1000 : 2200;
  const PULSE_MS = 1900;
  const stars = Array.from({ length: 150 }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: 0.5 + Math.random() * 1.6,
    a: 0.12 + Math.random() * 0.5,
  }));
  const earthImage = new Image();
  earthImage.onload = () => {
    state.earthImageReady = true;
  };
  earthImage.src = "assets/earth/earth_daymap_1024.jpg";

  state.rotationOffset = -TARGET.lon * Math.PI / 180;
  document.body.classList.add("intro-active");
  document.body.dataset.introRenderer = "canvas";
  roomPage?.setAttribute("aria-hidden", "true");
  resizeCanvas();
  state.routePoints = buildRoutePoints(state.visitor, TARGET);
  updateEnterLabel(state.visitor);

  let frameId = 0;

  enterButton.addEventListener("click", () => closeIntro());
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("orientationchange", resizeCanvas);

  Promise.race([
    fetchVisitorLocation(),
    new Promise((resolve) => setTimeout(() => resolve(null), 2600)),
  ]).then((location) => {
    if (state.closed || !location) return;
    state.visitor = location;
    state.routePoints = buildRoutePoints(location, TARGET);
    updateEnterLabel(location);
  }).catch(() => {});

  animate(performance.now());

  async function fetchVisitorLocation() {
    if (["localhost", "127.0.0.1", "::1"].includes(window.location.hostname)) {
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

  function resizeCanvas() {
    const rect = overlay.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.width = Math.max(1, Math.round(rect.width));
    state.height = Math.max(1, Math.round(rect.height));
    state.dpr = dpr;
    canvas.width = Math.round(state.width * dpr);
    canvas.height = Math.round(state.height * dpr);
    canvas.style.width = `${state.width}px`;
    canvas.style.height = `${state.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function updateEnterLabel(location) {
    if (!enterLabel) return;
    const miles = Math.round(distanceMiles(location, TARGET)).toLocaleString("en-US");
    enterLabel.textContent = `Welcome to visit Yu Guo from ${miles} miles away`;
  }

  function distanceMiles(start, end) {
    const earthRadiusMiles = 3958.8;
    const lat1 = start.lat * Math.PI / 180;
    const lat2 = end.lat * Math.PI / 180;
    const dLat = lat2 - lat1;
    const dLon = (end.lon - start.lon) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2
      + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
    return earthRadiusMiles * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  function normalize(vector) {
    const length = Math.hypot(vector.x, vector.y, vector.z) || 1;
    return { x: vector.x / length, y: vector.y / length, z: vector.z / length };
  }

  function geoToVector(lat, lon) {
    const latRad = lat * Math.PI / 180;
    const lonRad = lon * Math.PI / 180;
    return normalize({
      x: Math.cos(latRad) * Math.cos(lonRad),
      y: Math.sin(latRad),
      z: -Math.cos(latRad) * Math.sin(lonRad),
    });
  }

  function rotateY(vector, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return {
      x: vector.x * cos + vector.z * sin,
      y: vector.y,
      z: -vector.x * sin + vector.z * cos,
    };
  }

  function rotateX(vector, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return {
      x: vector.x,
      y: vector.y * cos - vector.z * sin,
      z: vector.y * sin + vector.z * cos,
    };
  }

  function project(vector, rotation, tilt, centerX, centerY, radius) {
    const rotated = rotateX(rotateY(vector, rotation), tilt);
    return {
      x: centerX + rotated.x * radius,
      y: centerY - rotated.y * radius,
      z: rotated.z,
    };
  }

  function buildRoutePoints(start, end, segments = 110) {
    const a = geoToVector(start.lat, start.lon);
    const b = geoToVector(end.lat, end.lon);
    const omega = Math.acos(clamp(dot(a, b), -1, 1));
    if (omega < 1e-5) return [a, b];
    const sinOmega = Math.sin(omega);
    const points = [];
    for (let i = 0; i < segments; i += 1) {
      const t = i / (segments - 1);
      const s1 = Math.sin((1 - t) * omega) / sinOmega;
      const s2 = Math.sin(t * omega) / sinOmega;
      points.push(normalize({
        x: a.x * s1 + b.x * s2,
        y: a.y * s1 + b.y * s2,
        z: a.z * s1 + b.z * s2,
      }));
    }
    return points;
  }

  function drawBackground(width, height) {
    const background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, "#020304");
    background.addColorStop(1, "#080b0f");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    stars.forEach((star, index) => {
      const twinkle = 0.55 + 0.45 * Math.sin((performance.now() / 800) + index * 1.37);
      ctx.fillStyle = `rgba(255, 255, 255, ${star.a * twinkle})`;
      ctx.beginPath();
      ctx.arc(star.x * width, star.y * height, star.r, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();

    ctx.save();
    const scan = ctx.createLinearGradient(0, 0, width, 0);
    scan.addColorStop(0, "rgba(117, 221, 205, 0)");
    scan.addColorStop(0.5, "rgba(117, 221, 205, 0.03)");
    scan.addColorStop(1, "rgba(117, 221, 205, 0)");
    ctx.fillStyle = scan;
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  }

  function drawSphere(centerX, centerY, radius, rotation, tilt) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.clip();

    const sphere = ctx.createRadialGradient(
      centerX - radius * 0.3,
      centerY - radius * 0.32,
      radius * 0.08,
      centerX,
      centerY,
      radius,
    );
    sphere.addColorStop(0, "rgba(47, 98, 131, 0.98)");
    sphere.addColorStop(0.45, "rgba(15, 50, 79, 0.98)");
    sphere.addColorStop(0.77, "rgba(7, 20, 32, 0.99)");
    sphere.addColorStop(1, "rgba(2, 8, 14, 1)");
    ctx.fillStyle = sphere;
    ctx.fillRect(centerX - radius, centerY - radius, radius * 2, radius * 2);

    if (state.earthImageReady) {
      const textureWidth = radius * 2.55;
      const textureHeight = radius * 2.02;
      const turn = (((rotation / (Math.PI * 2)) % 1) + 1) % 1;
      const offset = turn * textureWidth;
      const y = centerY - textureHeight * 0.5;
      ctx.globalAlpha = 0.98;
      for (let i = -1; i <= 1; i += 1) {
        ctx.drawImage(
          earthImage,
          centerX - radius - offset + i * textureWidth,
          y,
          textureWidth,
          textureHeight,
        );
      }
      ctx.globalAlpha = 1;
    }

    ctx.globalCompositeOperation = "screen";
    const glow = ctx.createRadialGradient(
      centerX - radius * 0.22,
      centerY - radius * 0.26,
      radius * 0.08,
      centerX,
      centerY,
      radius,
    );
    glow.addColorStop(0, "rgba(117, 221, 205, 0.12)");
    glow.addColorStop(0.45, "rgba(117, 221, 205, 0.04)");
    glow.addColorStop(1, "rgba(117, 221, 205, 0)");
    ctx.fillStyle = glow;
    ctx.fillRect(centerX - radius, centerY - radius, radius * 2, radius * 2);
    ctx.globalCompositeOperation = "source-over";

    const latitudes = [-60, -30, 0, 30, 60];
    const longitudes = [-150, -90, -30, 30, 90, 150];
    const gridColor = "rgba(174, 222, 255, 0.12)";
    const gridBright = "rgba(255, 255, 255, 0.18)";

    ctx.lineWidth = Math.max(0.8, radius * 0.0055);
    latitudes.forEach((lat) => {
      ctx.beginPath();
      let drawing = false;
      for (let lon = -180; lon <= 180; lon += 4) {
        const projected = project(geoToVector(lat, lon), rotation, tilt, centerX, centerY, radius);
        if (projected.z <= -0.1) {
          drawing = false;
          continue;
        }
        if (!drawing) {
          ctx.moveTo(projected.x, projected.y);
          drawing = true;
        } else {
          ctx.lineTo(projected.x, projected.y);
        }
      }
      ctx.strokeStyle = lat === 0 ? gridBright : gridColor;
      ctx.stroke();
    });

    longitudes.forEach((lon) => {
      ctx.beginPath();
      let drawing = false;
      for (let lat = -80; lat <= 80; lat += 3) {
        const projected = project(geoToVector(lat, lon), rotation, tilt, centerX, centerY, radius);
        if (projected.z <= -0.1) {
          drawing = false;
          continue;
        }
        if (!drawing) {
          ctx.moveTo(projected.x, projected.y);
          drawing = true;
        } else {
          ctx.lineTo(projected.x, projected.y);
        }
      }
      ctx.strokeStyle = gridColor;
      ctx.stroke();
    });

    ctx.restore();

    const rim = ctx.createRadialGradient(
      centerX - radius * 0.12,
      centerY - radius * 0.14,
      radius * 0.76,
      centerX,
      centerY,
      radius * 1.08,
    );
    rim.addColorStop(0, "rgba(255,255,255,0)");
    rim.addColorStop(0.7, "rgba(117, 221, 205, 0.02)");
    rim.addColorStop(1, "rgba(255,255,255,0.12)");
    ctx.strokeStyle = rim;
    ctx.lineWidth = Math.max(1.2, radius * 0.014);
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  function drawRoute(centerX, centerY, radius, rotation, tilt, progress, timeMs) {
    const projected = state.routePoints.map((point) => project(point, rotation, tilt, centerX, centerY, radius));
    const stop = Math.max(2, Math.min(projected.length, Math.round(progress * (projected.length - 1)) + 1));

    function drawStroke(lineWidth, alpha, blur) {
      ctx.save();
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowColor = "rgba(117, 221, 205, 0.9)";
      ctx.shadowBlur = blur;
      for (let i = 0; i < stop - 1; i += 1) {
        const a = projected[i];
        const b = projected[i + 1];
        const visibility = clamp((Math.min(a.z, b.z) + 0.22) / 0.42, 0, 1);
        if (visibility <= 0) continue;
        const segmentT = i / Math.max(1, projected.length - 2);
        const fade = 0.55 + 0.45 * (1 - segmentT);
        ctx.strokeStyle = `rgba(117, 221, 205, ${visibility * alpha * fade})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius + 1, 0, Math.PI * 2);
    ctx.clip();
    drawStroke(Math.max(2.8, radius * 0.021), 0.24, radius * 0.09);
    drawStroke(Math.max(1.6, radius * 0.009), 0.92, radius * 0.02);

    if (stop > 1) {
      const pulseIndex = Math.min(stop - 1, Math.floor(((timeMs % PULSE_MS) / PULSE_MS) * (stop - 1)));
      const pulse = projected[pulseIndex];
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.fillStyle = "rgba(246, 255, 252, 0.96)";
      ctx.shadowColor = "rgba(117, 221, 205, 0.9)";
      ctx.shadowBlur = radius * 0.14;
      ctx.beginPath();
      ctx.arc(pulse.x, pulse.y, Math.max(1.8, radius * 0.015), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.restore();
  }

  function drawPins(centerX, centerY, radius, rotation, tilt) {
    const source = project(geoToVector(state.visitor.lat, state.visitor.lon), rotation, tilt, centerX, centerY, radius);
    const target = project(geoToVector(TARGET.lat, TARGET.lon), rotation, tilt, centerX, centerY, radius);
    drawPin(source, "rgba(117, 221, 205, 0.95)", "rgba(117, 221, 205, 0.14)");
    drawLocationPin(target);
  }

  function drawPin(point, fillColor, glowColor) {
    if (point.z <= -0.12) return;
    const pinSize = Math.max(5, Math.min(11, state.width * 0.007));
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = glowColor;
    ctx.shadowColor = fillColor;
    ctx.shadowBlur = pinSize * 3;
    ctx.beginPath();
    ctx.arc(point.x, point.y, pinSize * 1.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.arc(point.x, point.y, pinSize * 0.7, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawLocationPin(point) {
    if (point.z <= -0.12) return;
    const pinSize = Math.max(16, Math.min(30, state.width * 0.018));
    ctx.save();
    ctx.translate(point.x, point.y - pinSize * 0.5);
    ctx.globalCompositeOperation = "lighter";
    ctx.shadowColor = "rgba(240, 179, 91, 0.75)";
    ctx.shadowBlur = pinSize * 1.2;
    ctx.fillStyle = "rgba(240, 179, 91, 0.26)";
    ctx.beginPath();
    ctx.arc(0, 0, pinSize * 0.88, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalCompositeOperation = "source-over";
    const gradient = ctx.createLinearGradient(0, -pinSize, 0, pinSize * 1.5);
    gradient.addColorStop(0, "#ffe2a3");
    gradient.addColorStop(0.55, "#f0b35b");
    gradient.addColorStop(1, "#ff6f4a");
    ctx.fillStyle = gradient;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = Math.max(2, pinSize * 0.13);
    ctx.beginPath();
    ctx.moveTo(0, pinSize * 1.34);
    ctx.bezierCurveTo(-pinSize * 0.14, pinSize * 0.78, -pinSize * 0.82, pinSize * 0.34, -pinSize * 0.82, -pinSize * 0.18);
    ctx.bezierCurveTo(-pinSize * 0.82, -pinSize * 0.72, -pinSize * 0.46, -pinSize * 1.1, 0, -pinSize * 1.1);
    ctx.bezierCurveTo(pinSize * 0.46, -pinSize * 1.1, pinSize * 0.82, -pinSize * 0.72, pinSize * 0.82, -pinSize * 0.18);
    ctx.bezierCurveTo(pinSize * 0.82, pinSize * 0.34, pinSize * 0.14, pinSize * 0.78, 0, pinSize * 1.34);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.arc(0, -pinSize * 0.2, pinSize * 0.28, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.84)";
    ctx.lineWidth = Math.max(1.5, pinSize * 0.08);
    ctx.beginPath();
    ctx.arc(0, -pinSize * 0.2, pinSize * 0.28, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function animate(now) {
    if (state.closed) return;
    const elapsed = now - START_TIME;
    const width = state.width;
    const height = state.height;
    const compact = width < 860 || height < 620;
    const globeX = width * 0.5;
    const globeY = height * (compact ? 0.49 : 0.47);
    const radius = Math.min(width, height) * (compact ? 0.27 : 0.31);
    const tilt = 0.36;
    const rotation = state.rotationOffset + elapsed * (compact ? 0.00003 : 0.000022);

    drawBackground(width, height);
    drawSphere(globeX, globeY, radius, rotation, tilt);
    const routeProgress = clamp(elapsed / ROUTE_DRAW_MS, 0, 1);
    drawRoute(globeX, globeY, radius, rotation, tilt, routeProgress, elapsed);
    drawPins(globeX, globeY, radius, rotation, tilt);

    frameId = window.requestAnimationFrame(animate);
  }

  function closeIntro() {
    if (state.closed) return;
    state.closed = true;
    window.cancelAnimationFrame(frameId);
    overlay.classList.add("is-hidden");
    document.body.classList.remove("intro-active");
    roomPage?.removeAttribute("aria-hidden");
    window.setTimeout(() => overlay.remove(), 560);
  }
})();
