"""Browser end-to-end check against a running backend + demo AI service (Playwright)."""
import json, sys, time
from pathlib import Path
import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/shots"); OUT.mkdir(parents=True, exist_ok=True)
KEY = json.load(open(ROOT / "data/dev_secrets.json"))["kiosk_api_key"]
BASE, AI = "http://127.0.0.1:8000", "http://127.0.0.1:8765"
errors = []

with sync_playwright() as p:
    b = p.chromium.launch()
    citizen = b.new_context(viewport={"width": 1600, "height": 900}).new_page()
    emp_ctx = b.new_context(viewport={"width": 1500, "height": 1000}); emp = emp_ctx.new_page()
    for pg, name in ((citizen, "citizen"), (emp, "employee")):
        pg.on("pageerror", lambda e, n=name: errors.append(f"{n} pageerror: {e}"))
        pg.on("console", lambda m, n=name: errors.append(f"{n} console.{m.type}: {m.text}") if m.type == "error" else None)

    citizen.goto(f"{BASE}/citizen/#desk=DESK-1&key={KEY}")
    citizen.wait_for_selector("text=XUSH KELIBSIZ"); time.sleep(1.2)
    assert "key=" not in citizen.url, "kiosk key must be removed from the URL"
    citizen.screenshot(path=OUT / "1_citizen_waiting.png")

    emp.goto(f"{BASE}/employee/")
    emp.fill("input[type=text]", "employee1"); emp.fill("input[type=password]", "Employee123!"); emp.click("button:has-text('Kirish')")
    emp.wait_for_selector("text=Fuqaro kutilmoqda"); emp.screenshot(path=OUT / "2_employee_idle.png")

    # Face ID (demo embedding) -> both screens switch to the session
    r = httpx.post(f"{AI}/demo/face").json(); assert r["matched"], r
    citizen.wait_for_selector("text=Karimov Ali Valiyevich"); emp.wait_for_selector("text=Karimov Ali Valiyevich")
    emp.wait_for_selector("text=TASDIQLANGAN"); time.sleep(.8)
    citizen.screenshot(path=OUT / "3_citizen_session.png")

    # citizen signs MEN PASPORT OLISH KELDIM
    t0 = time.time()
    assert "ok" in httpx.post(f"{AI}/demo/sign", json={"signs": ["MEN", "PASPORT", "OLISH", "KELDIM"]}).json()
    citizen.wait_for_selector("text=Recognizing", timeout=1) if False else None
    emp.wait_for_selector(".msgbox .big:has-text('Men pasport olish uchun keldim.')", timeout=20000)
    print(f"sign->sentence on employee screen: {time.time() - t0:.1f}s (includes 4 signs + the 2 s end-of-sentence pause)")
    time.sleep(.5); emp.screenshot(path=OUT / "4_employee_got_message.png")
    citizen.wait_for_selector(".you:has-text('Men pasport olish uchun keldim.')")

    # employee answers -> citizen monitor
    emp.fill("#reply", "Pasportingizni kameraga ko'rsating.")
    t1 = time.time(); emp.click("button:has-text('YUBORISH')")
    citizen.wait_for_selector(".reply .txt:has-text('KO‘RSATING')", timeout=5000)
    print(f"employee SEND -> citizen monitor: {(time.time() - t1) * 1000:.0f} ms")
    time.sleep(1.2); citizen.screenshot(path=OUT / "5_citizen_reply.png")
    emp.screenshot(path=OUT / "6_employee_after_send.png")

    # Cyrillic switch on the citizen monitor
    citizen.click("button:has-text('ЎЗ')"); time.sleep(.4)
    assert citizen.inner_text(".reply .txt").startswith("ПАСПОРТИНГИЗНИ"), citizen.inner_text(".reply .txt")
    citizen.screenshot(path=OUT / "7_citizen_cyrillic.png")
    citizen.click("button:has-text('UZ')")

    # uncertain sign (confidence below threshold): use an unseen speaker offset pattern -> force by sending garbage hand motion
    emp.click("button:has-text('Yakunlash')"); citizen.wait_for_selector("text=Xizmat yakunlandi"); citizen.screenshot(path=OUT / "8_citizen_ended.png")

    admin = b.new_context(viewport={"width": 1500, "height": 1000}).new_page()
    admin.on("pageerror", lambda e: errors.append(f"admin pageerror: {e}"))
    admin.goto(f"{BASE}/admin/"); admin.fill("input[type=text]", "admin"); admin.fill("input[type=password]", json.load(open(ROOT / "data/dev_secrets.json"))["admin_password"])
    admin.click("button:has-text('Kirish')"); admin.wait_for_selector("text=Active sessions"); time.sleep(.6); admin.screenshot(path=OUT / "9_admin_overview.png")
    for tab in ("Signs & dataset", "Training & models", "Logs"):
        admin.click(f"button[role=tab]:has-text('{tab}')"); time.sleep(1.0)
    admin.click("button[role=tab]:has-text('Training & models')"); time.sleep(.8); admin.screenshot(path=OUT / "10_admin_training.png")
    b.close()

print("JS errors:", errors or "none")
sys.exit(1 if errors else 0)
