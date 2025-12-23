// dashboard_ui.js
// 실시간 시계
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleString();
    document.getElementById("live-clock").innerText = timeStr;
}
setInterval(updateClock, 1000);
updateClock();

// 화면 전환 기능
function changeView(view) {
    const views = ["status", "growth", "disease", "device"];
    views.forEach(v => {
        const el = document.getElementById(`content-${v}`);
        el.style.display = (v === view) ? "block" : "none";
    });
}

function addLogItem(data) {
    const logList = document.getElementById("log-list");

    const log = document.createElement("div");
    log.classList.add("log-item");

    log.innerHTML = `
        <div class="log-time">${data.timestamp}</div>
        <div class="log-data">
            온도: ${data.temperature} °C,
            습도: ${data.humidity} %,
            CO₂: ${data.co2} ppm
        </div>
    `;

    logList.prepend(log);
}

async function loadAllLogs() {
    try {
        const res = await fetch("/api/data/all");
        const logs = await res.json();

        const logList = document.getElementById("log-list");
        logList.innerHTML = ""; // 기존 로그 비우기

        // logs는 오래된 순이므로 → 최신이 위로 가려면 뒤에서부터 추가
        for (let i = logs.length - 1; i >= 0; i--) {
            addLogItem(logs[i]);
        }

    } catch (err) {
        console.log("전체 로그 불러오기 실패:", err);
    }
}

// 처음 페이지 열 때 전체 로그 불러오기
loadAllLogs();

setInterval(loadLatestSensorData, 10000);