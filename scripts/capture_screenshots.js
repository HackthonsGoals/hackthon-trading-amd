const fs = require("fs");
const http = require("http");
const path = require("path");

const APP_URL = process.env.APP_URL || "http://127.0.0.1:8507";
const DEBUG_URL = process.env.EDGE_DEBUG_URL || "http://127.0.0.1:9231/json";
const OUT_DIR = path.resolve(__dirname, "..", "assets", "screenshots");

function getJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (response) => {
      let body = "";
      response.on("data", (chunk) => {
        body += chunk;
      });
      response.on("end", () => {
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on("error", reject);
    req.setTimeout(3000, () => req.destroy(new Error("request timed out")));
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const tabs = await getJson(DEBUG_URL);
  const tab = tabs.find((item) => item.type === "page" && item.url === "about:blank") || tabs.find((item) => item.type === "page");
  if (!tab) {
    throw new Error("No Edge page target found");
  }

  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  };

  await new Promise((resolve) => {
    ws.onopen = resolve;
  });

  function send(method, params = {}) {
    return new Promise((resolve) => {
      const message = { id: ++id, method, params };
      pending.set(message.id, resolve);
      ws.send(JSON.stringify(message));
    });
  }

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1400,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send("Page.navigate", { url: APP_URL });

  let ready = false;
  for (let index = 0; index < 90; index += 1) {
    await delay(1000);
    const result = await send("Runtime.evaluate", {
      expression: "document.body && document.body.innerText.includes('AMD GPU-Accelerated AI Signal Pipeline')",
    });
    ready = Boolean(result.result.result.value);
    if (ready) {
      break;
    }
  }
  if (!ready) {
    throw new Error("Dashboard text did not become ready");
  }

  async function screenshot(name) {
    const result = await send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
      fromSurface: true,
    });
    fs.writeFileSync(path.join(OUT_DIR, name), Buffer.from(result.result.data, "base64"));
  }

  await send("Runtime.evaluate", { expression: "window.scrollTo(0, 0)" });
  await delay(1500);
  await screenshot("dashboard-overview.png");

  await send("Runtime.evaluate", {
    expression:
      "(() => { const all = [...document.querySelectorAll('h1,h2,h3,[data-testid=stMarkdownContainer]')]; const el = all.find((item) => /Headline Sentiment/i.test(item.innerText || '')); if (el) el.scrollIntoView({ block: 'start' }); })()",
  });
  await delay(2000);
  await screenshot("sentiment-panel.png");

  await send("Runtime.evaluate", {
    expression:
      "(() => { const all = [...document.querySelectorAll('h1,h2,h3,[data-testid=stMarkdownContainer]')]; const el = all.find((item) => /Performance Metrics/i.test(item.innerText || '')); if (el) el.scrollIntoView({ block: 'start' }); })()",
  });
  await delay(2000);
  await screenshot("gpu-benchmark.png");

  ws.close();
  console.log("Captured dashboard-overview.png, sentiment-panel.png, gpu-benchmark.png");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
