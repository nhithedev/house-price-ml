/* Giao diện demo dự đoán giá nhà.
 * - Dropdown lấy từ options.json, bảng mô hình từ metrics.json (cả hai sinh bởi `python -m src.make_options`).
 * - Dự đoán: POST /api/predict. Nếu server chưa chạy thì dùng ước tính tạm theo trung vị khu vực
 *   (gắn nhãn rõ) để vẫn xem được giao diện.
 *
 * Hợp đồng API
 *   POST /api/predict  body: {tinh, quan_huyen, phuong_xa, dien_tich, so_phong_ngu, so_tang, so_phong_tam,
 *                             mat_tien, duong_vao, phap_ly, noi_that, huong_nha, la_du_an}  (trường tuỳ chọn có thể null)
 *                      trả về: {gia_m2, model, low?, high?}
 *   GET  /api/models   trả về: [{name, mae, rmse, r2, mape}]   (thiếu thì dùng metrics.json tĩnh)
 */
const $ = id => document.getElementById(id);
const API = "/api";
const MIN_N = 10;            // số tin tối thiểu để dùng trung vị của một khu vực làm mốc
const FEW = 100;             // tỉnh có ít hơn số tin này thì cảnh báo
let OPT;

const fmt = (n, d = 1) => n.toLocaleString("vi-VN", {maximumFractionDigits: d, minimumFractionDigits: d});
const opt = (v, label = v, sel = false) => `<option value="${v}"${sel ? " selected" : ""}>${label}</option>`;
const unknown = `<option value="">Không rõ</option>`;

async function init() {
  OPT = await (await fetch("options.json")).json();
  $("tinh").innerHTML = Object.entries(OPT.provinces)
    .map(([t, v]) => opt(t, `${t} (${v.count.toLocaleString("vi-VN")} tin)`, t === "Hà Nội")).join("");
  $("phaply").innerHTML = unknown + OPT.legal.map(v => opt(v)).join("");
  $("noithat").innerHTML = unknown + OPT.furniture.map(v => opt(v)).join("");
  $("huong").innerHTML = unknown + OPT.directions.map(v => opt(v)).join("");
  $("pn").innerHTML = unknown + Array.from({length: 9}, (_, i) => i + 1).map(n => opt(n, n, n === 3)).join("");
  $("dt").value = Math.round(OPT.area.median);
  fillDistricts();
  $("tinh").onchange = fillDistricts;
  $("quan").onchange = fillWards;
  $("form").onsubmit = onSubmit;
  loadModels();
}

function fillDistricts() {
  const p = OPT.provinces[$("tinh").value];
  $("quan").innerHTML = Object.entries(p.districts)
    .map(([q, v]) => opt(q, `${q} (${v.count})`)).join("");
  const note = $("tinhNote");
  note.hidden = p.count >= FEW;
  note.textContent = `Tỉnh này chỉ có ${p.count} tin trong dữ liệu huấn luyện, kết quả kém tin cậy.`;
  fillWards();
}

function fillWards() {
  const d = OPT.provinces[$("tinh").value].districts[$("quan").value];
  $("phuong").innerHTML = unknown + Object.entries(d.wards).map(([w, v]) => opt(w, `${w} (${v.count})`)).join("");
}

const num = id => $(id).value === "" ? null : +$(id).value;
const str = id => $(id).value || null;

function payload() {
  return {
    tinh: $("tinh").value, quan_huyen: $("quan").value, phuong_xa: str("phuong"),
    dien_tich: num("dt"), so_phong_ngu: num("pn"), so_tang: num("tang"), so_phong_tam: num("pt"),
    mat_tien: num("mt"), duong_vao: num("dv"), phap_ly: str("phaply"), noi_that: str("noithat"),
    huong_nha: str("huong"), la_du_an: $("duan").checked ? 1 : 0,
  };
}

/* Mốc so sánh: phường (đủ tin) -> quận -> tỉnh. */
function baseline(p) {
  const prov = OPT.provinces[p.tinh], d = prov.districts[p.quan_huyen], w = p.phuong_xa && d.wards[p.phuong_xa];
  if (w && w.count >= MIN_N) return {value: w.median, label: p.phuong_xa};
  if (d.count >= MIN_N) return {value: d.median, label: p.quan_huyen};
  return {value: prov.median, label: p.tinh};
}

async function predict(p) {
  let r;
  try {
    r = await fetch(`${API}/predict`, {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(p)});
  } catch {   // không kết nối được server -> ước tính tạm để vẫn xem được giao diện
    return {gia_m2: baseline(p).value, model: "Ước tính tạm (trung vị khu vực, chưa có server)"};
  }
  if (r.status === 404 || r.status === 501)   // đang mở bằng web server tĩnh, không có API
    return {gia_m2: baseline(p).value, model: "Ước tính tạm (trung vị khu vực, chưa có server)"};
  if (!r.ok) {                                  // server có chạy nhưng từ chối dữ liệu: báo lỗi thật
    let msg = `Server trả lỗi ${r.status}`;
    try { const e = await r.json(); if (Array.isArray(e.detail)) msg = e.detail.map(x => `${x.loc.slice(1).join(".")}: ${x.msg}`).join("; "); else if (e.detail) msg = e.detail; } catch {}
    return {error: msg};
  }
  return await r.json();
}

async function onSubmit(e) {
  e.preventDefault();
  const btn = e.submitter; btn.disabled = true;
  const p = payload(), res = await predict(p);
  btn.disabled = false;
  if (res.error) {
    $("out").hidden = true; $("empty").hidden = false;
    $("empty").innerHTML = ""; const s = document.createElement("span");
    s.className = "err"; s.textContent = "Không dự đoán được: " + res.error; $("empty").appendChild(s);
    return;
  }

  const base = baseline(p), delta = (res.gia_m2 / base.value - 1) * 100;
  const total = res.gia_m2 * p.dien_tich / 1000;

  $("empty").hidden = true; $("out").hidden = false;
  $("rPrice").textContent = fmt(res.gia_m2);
  $("rTotal").textContent = fmt(total, 2);
  $("rRange").textContent = res.low != null ? `Khoảng ${fmt(res.low)} – ${fmt(res.high)} triệu/m²` : "";
  $("rDelta").textContent = (delta >= 0 ? "+" : "") + fmt(delta, 0) + "%";
  $("barMark").style.left = Math.max(2, Math.min(98, 50 + delta / 2)) + "%";
  $("rBase").textContent = `Trung vị ${base.label}: ${fmt(base.value)} triệu/m²`;
  $("rSource").textContent = "Nguồn: " + res.model;

  const warn = [];
  if (total > OPT.price_cap_ty)
    warn.push(`Dữ liệu huấn luyện chỉ có nhà đến ${fmt(OPT.price_cap_ty)} tỷ đồng, kết quả vượt mức này không đáng tin.`);
  if (OPT.provinces[p.tinh].count < FEW)
    warn.push(`${p.tinh} có rất ít dữ liệu.`);
  $("rWarn").hidden = !warn.length;
  $("rWarn").textContent = warn.join(" ");
}

async function loadModels() {
  let ms = null;
  try { const r = await fetch(`${API}/models`); if (r.ok) ms = await r.json(); } catch {}
  if (!ms) { try { const r = await fetch("metrics.json"); if (r.ok) ms = await r.json(); } catch {} }
  if (!ms || !ms.length) return;
  $("modelRows").innerHTML = ms.map(m => `<tr><td>${m.name}</td><td>${fmt(m.mae, 2)}</td>
    <td>${fmt(m.rmse, 2)}</td><td>${fmt(m.r2, 3)}</td><td>${fmt(m.mape, 1)}%</td></tr>`).join("");
  $("modelsNote").textContent = ms[0].split ? `(tính trên tập ${ms[0].split})` : "";
  $("models").hidden = false;
}

init().catch(() => {
  $("empty").innerHTML = '<span class="err">Không tải được options.json. Hãy chạy qua một web server (xem README).</span>';
});
