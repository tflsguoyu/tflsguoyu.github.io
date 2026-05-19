(async () => {
  try {
    await import("./intro-webgl.js?v=production-clean-1");
  } catch (error) {
    console.warn("WebGL intro failed, falling back to the canvas intro.", error);
    document.body.dataset.webglIntroError = error?.message || String(error);
    await import("./intro-canvas.js?v=production-clean-1");
  }
})();
